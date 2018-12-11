import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QGroupBox, \
    QFormLayout, QVBoxLayout, QDialog, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from model import predict


class App(QDialog, QWidget):
    def __init__(self):
        super().__init__()
        self.pix_label = QLabel(self)
        self.text_label = QLabel(self)
        self.main_layout = QVBoxLayout()
        self.formGroupBox = QGroupBox()
        self.test_image = ''
        self.title = 'Disease Predictor'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.create_form_group_box()

        self.main_layout.addWidget(self.formGroupBox)
        self.setLayout(self.main_layout)
        self.main_layout.addWidget(self.pix_label)
        self.main_layout.addWidget(self.text_label)
        self.show()

    def create_form_group_box(self):
        layout = QFormLayout()
        upload = QPushButton('Upload X-Ray Scan', self)
        upload.clicked.connect(self.open_file_name_dialog)
        layout.addRow(upload)
        self.formGroupBox.setLayout(layout)

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "File Directory", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        if file_name:
            self.test_image = file_name
            self.show_uploaded_image()

    def show_uploaded_image(self):
        pixmap = QPixmap(self.test_image)
        pixmap = pixmap.scaledToWidth(256, 256)
        self.pix_label.setPixmap(pixmap)
        check = predict(self.test_image)
        self.text_label.setText(",".join(check))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
