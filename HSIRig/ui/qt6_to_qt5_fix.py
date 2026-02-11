import re
import sys
import os

def downgrade_ui(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Cleaning {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # List of common Qt6 namespaces that break PyQt5
    namespaces = [
        "WindowModality",
        "Orientation",
        "LayoutDirection",
        "AlignmentFlag",
        "InputMethodHint",
        "ShortcutContext",
        "ScrollBarPolicy",
        "FocusPolicy",
        "ContextMenuPolicy",
        "ArrowType",
        "ToolButtonStyle",
        "FrameShape",
        "FrameShadow",
        "SizeAdjustmentPolicy",
        "TabPosition",
        "TabShape"
    ]

    # This regex looks for 'Qt::NamespaceName::' and replaces it with 'Qt::'
    # Example: 'Qt::AlignmentFlag::AlignLeft' -> 'Qt::AlignLeft'
    original_content = content
    for ns in namespaces:
        pattern = rf"Qt::{ns}::"
        content = re.sub(pattern, "Qt::", content)

    if content == original_content:
        print("No Qt6 specific namespaces found. File might already be clean.")
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully cleaned {file_path}.")

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "main_window.ui"
    downgrade_ui(target_file)