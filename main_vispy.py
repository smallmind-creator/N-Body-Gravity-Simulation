import sys
from PyQt5.QtWidgets import QApplication
from ui_vispy import NBodyVisPyApp

# --- Main Execution ---
if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    main_window = NBodyVisPyApp()
    main_window.show()
   
    sys.exit(app.exec_())
