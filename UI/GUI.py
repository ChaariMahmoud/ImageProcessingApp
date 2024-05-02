import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QFileDialog, QAction
from PyQt5.QtGui import QPixmap, QImage, QPainter
import cv2 as cv
import matplotlib.pyplot as plt
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 800, 600)

        # Widgets
        self.image_label = QLabel()
        self.contours_label = QLabel()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.contours_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Menu bar
        self.menu_bar = self.menuBar()
        self.image_menu = self.menu_bar.addMenu("Image")
        self.histogram_menu = self.menu_bar.addMenu("Histogram")
        self.filters_menu = self.menu_bar.addMenu("Filters")
        self.brightness_menu = self.menu_bar.addMenu("brightness")
        self.contrast_menu = self.menu_bar.addMenu("contrast")
        self.Other_Functions_menu = self.menu_bar.addMenu("Other Functions")

        # Image menu actions
        self.load_action = QAction("Load Image", self)
        self.load_action.triggered.connect(self.load_image)
        self.save_action = QAction("Save Image", self)
        self.save_action.triggered.connect(self.save_image)
        self.print_action = QAction("Print Image", self)
        self.print_action.triggered.connect(self.print_image)
        self.image_menu.addAction(self.load_action)
        self.image_menu.addAction(self.save_action)
        self.image_menu.addAction(self.print_action)

        # Histogram menu actions
        self.histogram_action = QAction("Compute Histogram", self)
        self.histogram_action.triggered.connect(self.histogram_calculation)
        self.color_histogram_action = QAction("Compute Color Histogram ", self)
        self.color_histogram_action.triggered.connect(
            self.compute_color_histogram)
        self.histogram_menu.addAction(self.histogram_action)
        self.histogram_menu.addAction(self.color_histogram_action)

        # Filters menu actions
        self.gray_action = QAction("Convert to grayscale", self)
        self.gray_action.triggered.connect(self.convert_to_grayscale)
        self.apply_filter_average_action = QAction(
            "apply average filter", self)
        self.apply_filter_average_action.triggered.connect(
            self.apply_filter_average)
        self.apply_filter_median_action = QAction("apply median filter", self)
        self.apply_filter_median_action.triggered.connect(
            self.apply_filter_median)
        self.apply_filter_min_action = QAction("apply min filter", self)
        self.apply_filter_min_action.triggered.connect(self.apply_filter_min)
        self.apply_filter_max_action = QAction("apply max filter", self)
        self.apply_filter_max_action.triggered.connect(self.apply_filter_max)
        self.apply_filter_canny_edges_action = QAction(
            "apply canny edges filter", self)
        self.apply_filter_canny_edges_action.triggered.connect(
            self.apply_filter_canny_edges)
        self.filters_menu.addAction(self.gray_action)
        self.filters_menu.addAction(self.apply_filter_average_action)
        self.filters_menu.addAction(self.apply_filter_median_action)
        self.filters_menu.addAction(self.apply_filter_min_action)
        self.filters_menu.addAction(self.apply_filter_max_action)
        self.filters_menu.addAction(self.apply_filter_canny_edges_action)

        # Brightness menu actions
        self.increase_brightness_action = QAction(
            "+ Increase brightness", self)
        self.increase_brightness_action.triggered.connect(
            self.increase_brightness)
        self.decrease_brightness_action = QAction(
            "- Decrease brightness", self)
        self.decrease_brightness_action.triggered.connect(
            self.decrease_brightness)
        self.brightness_menu.addAction(self.increase_brightness_action)
        self.brightness_menu.addAction(self.decrease_brightness_action)

        # Contrast menu actions
        self.increase_contrast_action = QAction("+ Increase contrast", self)
        self.increase_contrast_action.triggered.connect(self.increase_contrast)
        self.decrease_contrast_action = QAction("- Decrease contrast", self)
        self.decrease_contrast_action.triggered.connect(self.decrease_contrast)
        self.contrast_menu.addAction(self.increase_contrast_action)
        self.contrast_menu.addAction(self.decrease_contrast_action)

        # Other functions menu actions
        self.contours_detection_action = QAction("Contours detection", self)
        self.contours_detection_action.triggered.connect(
            self.contours_detection)
        self.segment_image_action = QAction("Segment image", self)
        self.segment_image_action.triggered.connect(
            self.segment_image)
        self.Other_Functions_menu.addAction(self.contours_detection_action)
        self.Other_Functions_menu.addAction(self.segment_image_action)

        # Initialize variables
        self.image = None

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image", "images", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image = cv.imread(file_path)
            self.display_image()

    def display_image(self):
        if self.image is not None:
            width = self.image.shape[1]
            height = self.image.shape[0]
            scale_factor = 3  # To adjust this factor to increase/decrease the size

            # Resizing the image using OpenCV
            reseized_image = cv.resize(
                self.image, (width * scale_factor, height * scale_factor), interpolation=cv.INTER_CUBIC)

            # Convert the resized image to a QImage
            q_img = QImage(reseized_image.data, reseized_image.shape[1], reseized_image.shape[0],
                           reseized_image.strides[0], QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            # scaled_pixmap = pixmap.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.image_label.show()

    def convert_to_grayscale(self):
        if self.image is not None:
            gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            self.image = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
            self.display_image()

    def increase_contrast(self):
        if self.image is not None:
            alpha = 1.2  # Increase contrast factor (1-3)
            beta = 0    # Keep brightness unchanged (0-100)
            self.image = cv.convertScaleAbs(self.image, alpha=alpha, beta=beta)
            self.display_image()

    def decrease_contrast(self):
        if self.image is not None:
            alpha = 0.8  # Decrease contrast factor
            beta = 0     # Keep brightness unchanged
            self.image = cv.convertScaleAbs(self.image, alpha=alpha, beta=beta)
            self.display_image()

    def increase_brightness(self):
        if self.image is not None:
            alpha = 1  # Increase contrast factor (1-3)
            beta = 10     # Keep brightness unchanged (0-100)
            self.image = cv.convertScaleAbs(self.image, alpha=alpha, beta=beta)
            self.display_image()

    def decrease_brightness(self):
        if self.image is not None:
            alpha = 1  # Decrease contrast factor
            beta = -10     # Keep brightness unchanged
            self.image = cv.convertScaleAbs(self.image, alpha=alpha, beta=beta)
            self.display_image()

    def apply_filter_average(self):
        if self.image is not None:
            self.image = cv.blur(self.image, (3, 3))
            self.display_image()

    def apply_filter_median(self):
        if self.image is not None:
            self.image = cv.medianBlur(self.image, 3)
            self.display_image()

    def apply_filter_min(self):
        if self.image is not None:
            self.image = cv.erode(self.image, (7, 7), None, iterations=3)
            self.display_image()

    def apply_filter_max(self):
        if self.image is not None:
            self.image = cv.dilate(self.image, (7, 7), None, iterations=3)
            self.display_image()

    def apply_filter_canny_edges(self):
        if self.image is not None:
            edges = cv.Canny(self.image, 50, 150)
            self.image = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            self.display_image()

    def contours_detection(self):
        if self.image is not None:
            edges = cv.Canny(self.image, 50, 150)
            self.image = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            contours, hierarchies = cv.findContours(
                edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contours_count = "there are " + \
                str(len(contours))+" countours found in the image "
            self.contours_label.setText(contours_count)

    def histogram_calculation(self):
        if self.image is not None:
            gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
            plt.plot(hist)
            plt.xlim([0, 256])
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title('Histogram')
            plt.show()

    def compute_color_histogram(self):
        if self.image is not None:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
                plt.title('Color Histogram')
            plt.show()

    def save_image(self):
        if self.image is not None:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self, "Save Image", "", "Image Files (*.png *.jpg)")
            if file_path:
                cv.imwrite(file_path, self.image)

    def print_image(self):
        if self.image is not None:
            printer = QPrinter()
            dialog = QPrintDialog(printer, self)
            if dialog.exec_() == QPrintDialog.Accepted:
                painter = QPainter(printer)
                rect = painter.viewport()
                size = self.image_label.pixmap().size()
                size.scale(rect.size(), Qt.KeepAspectRatio)
                painter.setViewport(rect.x(), rect.y(),
                                    size.width(), size.height())
                painter.setWindow(self.image_label.pixmap().rect())
                painter.drawPixmap(0, 0, self.image_label.pixmap())
                
    #simple segmentation
    def segment_image(self):
        if self.image is not None:
            gray_image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            ret, segmented_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
            self.image = cv.cvtColor(segmented_image, cv.COLOR_GRAY2BGR)
            self.display_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
