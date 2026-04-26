import adsk.core, adsk.fusion, traceback, threading, time, json
try:
    from urllib.request import urlopen
    from urllib.error import URLError
except ImportError:
    pass

API_BASE = "http://172.17.62.117:5000/api"

def fetch_health_index(day):
    """Fetch HI for a specific day from the Flask API."""
    url = f"{API_BASE}/health-index?day={day}"
    try:
        with urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data
    except Exception as e:
        return None

def fetch_total_days():
    """Get total number of days available."""
    try:
        with urlopen(f"{API_BASE}/days", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return len(data)
    except:
        return None

def hi_to_rgb(hi):
    """
    Convert Health Index (0.0 = failure/red, 1.0 = healthy/green)
    through green → yellow → orange → red.
    """
    hi = max(0.0, min(1.0, hi))

    if hi >= 0.66:
        # Green to Yellow (hi: 1.0 → 0.66)
        t = (hi - 0.66) / 0.34
        r = int((1 - t) * 255)
        g = int(180 + t * 75)   # 255 → 180
        b = 0
    elif hi >= 0.33:
        # Yellow to Orange (hi: 0.66 → 0.33)
        t = (hi - 0.33) / 0.33
        r = 255
        g = int(t * 115 + 140)  # 140 → 255
        b = 0
    else:
        # Orange to Red (hi: 0.33 → 0.0)
        t = hi / 0.33
        r = 255
        g = int(t * 140)        # 0 → 140
        b = 0

    return r, g, b

def find_base_appearance(app):
    keywords = ['plastic', 'matte', 'paint', 'flat', 'opaque', 'steel', 'metal']
    for lib in app.materialLibraries:
        try:
            for a in lib.appearances:
                try:
                    if any(kw in a.name.lower() for kw in keywords):
                        return a
                except:
                    continue
        except:
            continue
    try:
        return app.materialLibraries.item(0).appearances.item(0)
    except:
        return None

def apply_color(app, design, body, r, g, b):
    try:
        app_name = 'IR_HI_Color'
        new_app = design.appearances.itemByName(app_name)
        if not new_app:
            base = find_base_appearance(app)
            if not base:
                return False
            new_app = design.appearances.addByCopy(base, app_name)

        for prop in new_app.appearanceProperties:
            try:
                if prop.objectType == adsk.core.ColorProperty.classType():
                    cp = adsk.core.ColorProperty.cast(prop)
                    cp.value = adsk.core.Color.create(r, g, b, 255)
                    break
            except:
                continue

        body.appearance = new_app
        adsk.doEvents()
        return True
    except:
        return False

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        root   = design.rootComponent

        # ── Find Inner Race body ──
        inner_race = None
        for body in root.bRepBodies:
            if body.name == 'Inner Race':
                inner_race = body
                break

        if not inner_race:
            ui.messageBox('Body named "Inner Race" not found.\nCheck the name in your browser panel.')
            return

        # ── Get total days from API ──
        total_days = fetch_total_days()
        if total_days is None:
            ui.messageBox('Could not connect to API at:\n' + API_BASE +
                          '\n\nMake sure your Flask server is running.')
            return

        # ── Ask user which mode they want ──
        mode_result = ui.messageBox(
            f'Bearing Health Visualiser\n'
            f'Total days available: {total_days}\n\n'
            f'Choose mode:\n'
            f'  [Yes]  → Animate through ALL days\n'
            f'  [No]   → Pick a specific day',
            'Bearing Health Visualiser',
            adsk.core.MessageBoxButtonTypes.YesNoCancelButtonType
        )

        if mode_result == adsk.core.DialogResults.DialogCancel:
            return

        animate_all = (mode_result == adsk.core.DialogResults.DialogYes)

        if animate_all:
            # ── Animate through all days ──
            def animate():
                try:
                    for day in range(total_days):
                        data = fetch_health_index(day)
                        if data is None:
                            continue

                        hi = data.get('current_hi', None)
                        if hi is None:
                            continue

                        r, g, b = hi_to_rgb(hi)
                        apply_color(app, design, inner_race, r, g, b)

                        status  = data.get('status', 'unknown').upper()
                        hi_pct  = round(hi * 100, 1)

                        # Update every day with a short pause
                        time.sleep(0.3)

                    # Final state message
                    final_data = fetch_health_index(total_days - 1)
                    if final_data:
                        final_hi  = final_data.get('current_hi', 0)
                        final_st  = final_data.get('status', 'unknown').upper()
                        rul       = final_data.get('rul_days')
                        rul_str   = f"{rul} days" if rul is not None else "N/A"
                        ui.messageBox(
                            f'Animation complete.\n\n'
                            f'Final Health Index : {round(final_hi * 100, 1)}%\n'
                            f'Status             : {final_st}\n'
                            f'RUL                : {rul_str}',
                            'Bearing Health — Final State'
                        )
                except:
                    ui.messageBox('Animation error:\n' + traceback.format_exc())

            t = threading.Thread(target=animate)
            t.daemon = True
            t.start()

        else:
            # ── Single day picker ──
            input_result, cancelled = ui.inputBox(
                f'Enter day number (0 to {total_days - 1}):',
                'Select Day',
                str(total_days - 1)   # default = last day
            )

            if cancelled:
                return

            try:
                day = int(input_result.strip())
                if not (0 <= day < total_days):
                    ui.messageBox(f'Day must be between 0 and {total_days - 1}.')
                    return
            except ValueError:
                ui.messageBox('Invalid input. Please enter a whole number.')
                return

            # Fetch and apply
            data = fetch_health_index(day)
            if data is None:
                ui.messageBox(f'Could not fetch data for day {day}.\nCheck the server.')
                return

            hi = data.get('current_hi', None)
            if hi is None:
                ui.messageBox(f'No health index returned for day {day}.')
                return

            r, g, b = hi_to_rgb(hi)
            apply_color(app, design, inner_race, r, g, b)

            status  = data.get('status', 'unknown').upper()
            rul     = data.get('rul_days')
            rul_str = f"{rul} days" if rul is not None else "N/A"
            ts      = data.get('latest_timestamp', 'N/A')

            ui.messageBox(
                f'Day {day} — {ts}\n\n'
                f'Health Index : {round(hi * 100, 1)}%\n'
                f'Status       : {status}\n'
                f'RUL          : {rul_str}\n\n'
                f'Color applied to Inner Race.',
                'Bearing Health Index'
            )

    except:
        if ui:
            ui.messageBox('Script error:\n' + traceback.format_exc())