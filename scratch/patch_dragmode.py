with open("src/terrain_mapper/web_sim.py", "r") as f:
    content = f.read()

callback_code = """
    @app.callback(
        Output("editor-graph", "figure", allow_duplicate=True),
        Input("edit-mode", "value"),
        prevent_initial_call=True,
    )
    def update_dragmode(mode: str) -> Patch:
        patched = Patch()
        patched["layout"]["dragmode"] = False if mode in ("Waypoint", "Spawn", "Goal") else "select"
        return patched

    return app"""

content = content.replace("    return app", callback_code)

with open("src/terrain_mapper/web_sim.py", "w") as f:
    f.write(content)
