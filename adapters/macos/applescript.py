"""AppleScript builder helpers — return script strings, never execute."""

from __future__ import annotations


def build_open_app_script(app_name: str) -> str:
    return f'tell application "{app_name}" to activate'


def build_switch_window_script(app_name: str, title_pattern: str | None = None) -> str:
    if title_pattern:
        return (
            f'tell application "System Events"\n'
            f'  tell process "{app_name}"\n'
            f'    set frontmost to true\n'
            f'    set targetWindow to first window whose name contains "{title_pattern}"\n'
            f'    perform action "AXRaise" of targetWindow\n'
            f'  end tell\n'
            f'end tell'
        )
    return build_open_app_script(app_name)


def build_notification_script(title: str, body: str) -> str:
    safe_title = title.replace('"', '\\"').replace("'", "\\'")
    safe_body = body.replace('"', '\\"').replace("'", "\\'")
    return f'display notification "{safe_body}" with title "{safe_title}"'


def build_open_url_script(url: str, browser: str = "Safari") -> str:
    return (
        f'tell application "{browser}"\n'
        f'  activate\n'
        f'  open location "{url}"\n'
        f'end tell'
    )


def build_say_script(text: str, voice: str = "Samantha") -> str:
    escaped = text.replace('"', '\\"')
    return f'do shell script "say -v {voice} \\"{escaped}\\""'


def build_play_audio_script(path: str) -> str:
    return f'do shell script "afplay \\"{path}\\"" '


def build_list_running_apps_script() -> str:
    return (
        'tell application "System Events"\n'
        '  set appList to name of every application process whose background only is false\n'
        'end tell\n'
        'return appList'
    )


def build_get_active_workspace_script() -> str:
    return (
        'tell application "System Events"\n'
        '  set frontApp to name of first application process whose frontmost is true\n'
        'end tell\n'
        'return frontApp'
    )
