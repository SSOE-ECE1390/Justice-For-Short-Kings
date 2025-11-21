"""
Justice For Short Kings - Main GUI Application
Integrates all team member features with webcam and image processing
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QFrame, QScrollArea,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy, QCheckBox, QLineEdit
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread, QRectF
from PyQt6.QtGui import QImage, QPixmap, QFont, QWheelEvent, QPainter
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class ZoomableImageViewer(QGraphicsView):
    """Interactive image viewer with zoom and pan capabilities."""

    def __init__(self):
        super().__init__()

        # Create scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Image item
        self.image_item = None
        self.current_pixmap = None

        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.GlobalColor.black)

        # Zoom settings
        self.zoom_factor = 1.15
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.current_zoom = 1.0

        # Style
        self.setStyleSheet("""
            QGraphicsView {
                border: 2px solid #3498db;
                border-radius: 5px;
            }
        """)

    def set_image(self, pixmap: QPixmap):
        """Set image to display."""
        self.current_pixmap = pixmap

        # Clear scene
        self.scene.clear()

        if pixmap is not None and not pixmap.isNull():
            # Add pixmap to scene
            self.image_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.image_item)

            # Fit image to view
            self.fit_to_view()

    def fit_to_view(self):
        """Fit image to view while maintaining aspect ratio."""
        if self.image_item is not None:
            self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.current_zoom = 1.0

    def zoom_in(self):
        """Zoom in."""
        if self.current_zoom < self.max_zoom:
            self.scale(self.zoom_factor, self.zoom_factor)
            self.current_zoom *= self.zoom_factor

    def zoom_out(self):
        """Zoom out."""
        if self.current_zoom > self.min_zoom:
            self.scale(1.0 / self.zoom_factor, 1.0 / self.zoom_factor)
            self.current_zoom /= self.zoom_factor

    def reset_zoom(self):
        """Reset zoom to fit view."""
        self.fit_to_view()

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset zoom."""
        self.reset_zoom()
        event.accept()


class ProcessingThread(QThread):
    """Thread for running image processing without freezing GUI."""
    finished = pyqtSignal(dict)  # Emits results dictionary
    error = pyqtSignal(str)  # Emits error message

    def __init__(self, feature_name, image, processor, equalize=False, replace=False, replace_prompt="", replace_method="sdxl"):
        super().__init__()
        self.feature_name = feature_name
        self.image = image
        self.processor = processor
        self.equalize = equalize
        self.replace = replace
        self.replace_prompt = replace_prompt
        self.replace_method = replace_method

    def run(self):
        """Run processing in background thread."""
        try:
            if self.feature_name == "Will - Segmentation Expansion":
                # Run Will's v2 processing
                results = self.processor.process_image(
                    self.image,
                    conf_threshold=0.5,
                    equalize=self.equalize,
                    replace=self.replace,
                    replace_prompt=self.replace_prompt,
                    replace_method=self.replace_method
                )
                self.finished.emit(results)
            else:
                # Placeholder for other features
                self.finished.emit({'result': self.image})
        except Exception as e:
            self.error.emit(str(e))


class MainGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Justice For Short Kings - Height Equalizer")
        self.setGeometry(100, 100, 1400, 900)

        # State
        self.webcam = None
        self.webcam_active = False
        self.current_frame = None
        self.current_image = None  # For uploaded images
        self.using_uploaded_image = False
        self.processor = None
        self.processing_thread = None
        self.processed_results = None

        # Initialize UI
        self.init_ui()

        # Timer for webcam updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Initialize webcam
        self.start_webcam()

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Top control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Main content area (webcam/image + processing results)
        content_layout = QHBoxLayout()

        # Left side: Main viewer
        self.viewer_frame = self.create_main_viewer()
        content_layout.addWidget(self.viewer_frame, stretch=3)

        # Right side: Processing pipeline (initially hidden)
        self.pipeline_frame = self.create_pipeline_viewer()
        self.pipeline_frame.setVisible(False)
        content_layout.addWidget(self.pipeline_frame, stretch=1)

        main_layout.addLayout(content_layout)

        # Status bar
        self.status_label = QLabel("Ready - Select a feature and capture/upload an image")
        self.status_label.setStyleSheet("padding: 5px; background-color: #2c3e50; color: white;")
        main_layout.addWidget(self.status_label)

    def create_control_panel(self):
        """Create the top control panel with buttons and dropdown."""
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: #34495e; border-radius: 5px; padding: 10px; }")
        panel.setMaximumHeight(80)

        layout = QHBoxLayout(panel)

        # Title
        title = QLabel("Justice For Short Kings")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        layout.addWidget(title)

        layout.addStretch()

        # Feature selector
        feature_label = QLabel("Feature:")
        feature_label.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(feature_label)

        self.feature_combo = QComboBox()
        self.feature_combo.addItems([
            "None - Live Webcam",
            "Will - Segmentation Expansion",
            "Iyan - Hat & Stretch",
            "Connor - Feature",
            "Yoel - Face Tracking",
            "Raymond - Feature"
        ])
        self.feature_combo.setMinimumWidth(250)
        self.feature_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
        """)
        self.feature_combo.currentTextChanged.connect(self.on_feature_changed)
        layout.addWidget(self.feature_combo)

        # Upload button
        self.upload_btn = QPushButton("📁 Upload Image")
        self.upload_btn.setStyleSheet(self.get_button_style("#3498db"))
        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn)

        # Capture button
        self.capture_btn = QPushButton("📷 Capture from Webcam")
        self.capture_btn.setStyleSheet(self.get_button_style("#2ecc71"))
        self.capture_btn.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_btn)

        # Process button
        self.process_btn = QPushButton("▶️ Process Image")
        self.process_btn.setStyleSheet(self.get_button_style("#e74c3c"))
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)

        # Equalize checkbox (for Will's feature)
        self.equalize_checkbox = QCheckBox("⚖ Equalize to Average Height")
        self.equalize_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.equalize_checkbox.setToolTip("Scale all people to average height instead of expanding shortest to tallest")
        self.equalize_checkbox.setVisible(False)  # Hidden by default
        layout.addWidget(self.equalize_checkbox)

        # Person replacement controls
        self.replace_checkbox = QCheckBox("🎭 Replace Shortest Person")
        self.replace_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.replace_checkbox.setToolTip("Replace the shortest person with AI-generated content")
        self.replace_checkbox.setVisible(False)
        self.replace_checkbox.stateChanged.connect(self.on_replace_toggled)
        layout.addWidget(self.replace_checkbox)

        # Replacement prompt input
        self.replace_prompt = QLineEdit()
        self.replace_prompt.setPlaceholderText("Enter replacement (e.g., 'kangaroo', 'ghost', 'Albert Einstein')")
        self.replace_prompt.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 5px;
                font-size: 13px;
            }
        """)
        self.replace_prompt.setVisible(False)
        layout.addWidget(self.replace_prompt)

        # Replacement method dropdown
        self.replace_method_combo = QComboBox()
        self.replace_method_combo.addItems([
            "SDXL Turbo (Fast)",
            "ControlNet (Pose-Aware)"
        ])
        self.replace_method_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 5px;
                font-size: 13px;
            }
        """)
        self.replace_method_combo.setToolTip("Choose replacement method: SDXL Turbo is faster, ControlNet preserves pose")
        self.replace_method_combo.setVisible(False)
        layout.addWidget(self.replace_method_combo)

        return panel

    def get_button_style(self, color):
        """Get button stylesheet."""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
            QPushButton:disabled {{
                background-color: #95a5a6;
            }}
        """

    def create_main_viewer(self):
        """Create the main image/webcam viewer."""
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #2c3e50; border-radius: 5px; }")

        layout = QVBoxLayout(frame)

        # Title bar with zoom controls
        title_layout = QHBoxLayout()

        title = QLabel("Main Viewer")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold; padding: 5px;")
        title_layout.addWidget(title)

        title_layout.addStretch()

        # Zoom controls
        zoom_out_btn = QPushButton("🔍−")
        zoom_out_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: 1px solid #3498db;
                border-radius: 3px;
                padding: 5px 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        zoom_out_btn.setToolTip("Zoom Out (or scroll down)")
        zoom_out_btn.clicked.connect(self.zoom_out)
        title_layout.addWidget(zoom_out_btn)

        zoom_reset_btn = QPushButton("🔍◯")
        zoom_reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: 1px solid #3498db;
                border-radius: 3px;
                padding: 5px 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        zoom_reset_btn.setToolTip("Reset Zoom (or double-click)")
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        title_layout.addWidget(zoom_reset_btn)

        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: 1px solid #3498db;
                border-radius: 3px;
                padding: 5px 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        zoom_in_btn.setToolTip("Zoom In (or scroll up)")
        zoom_in_btn.clicked.connect(self.zoom_in)
        title_layout.addWidget(zoom_in_btn)

        layout.addLayout(title_layout)

        # Zoomable image viewer
        self.main_viewer = ZoomableImageViewer()
        self.main_viewer.setMinimumSize(640, 480)
        layout.addWidget(self.main_viewer)

        # Help text
        help_text = QLabel("💡 Scroll to zoom • Drag to pan • Double-click to reset")
        help_text.setStyleSheet("color: #95a5a6; font-size: 11px; padding: 3px;")
        help_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(help_text)

        return frame

    def create_pipeline_viewer(self):
        """Create the processing pipeline viewer (for Will's feature)."""
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #34495e; border-radius: 5px; }")

        layout = QVBoxLayout(frame)

        # Title
        title = QLabel("Processing Pipeline")
        title.setStyleSheet("color: white; font-size: 14px; font-weight: bold; padding: 5px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Scroll area for pipeline images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #34495e; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Create pipeline step viewers
        self.pipeline_viewers = {}
        pipeline_steps = [
            ("step1_original", "1. Original"),
            ("step2_detections", "2. Detections"),
            ("step3_segmentations", "3. Segmentations"),
            ("step4_expansion", "4. Expansion"),
        ]

        for step_key, step_name in pipeline_steps:
            step_frame = QFrame()
            step_frame.setStyleSheet("QFrame { background-color: #2c3e50; border-radius: 3px; margin: 5px; }")
            step_layout = QVBoxLayout(step_frame)

            step_label = QLabel(step_name)
            step_label.setStyleSheet("color: white; font-size: 12px; padding: 3px;")
            step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            step_layout.addWidget(step_label)

            step_viewer = QLabel()
            step_viewer.setStyleSheet("background-color: black; border: 1px solid #3498db;")
            step_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            step_viewer.setMinimumHeight(150)
            step_viewer.setScaledContents(True)
            step_layout.addWidget(step_viewer)

            scroll_layout.addWidget(step_frame)
            self.pipeline_viewers[step_key] = step_viewer

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        return frame

    def start_webcam(self):
        """Start webcam capture."""
        self.webcam = cv2.VideoCapture(0)
        if self.webcam.isOpened():
            self.webcam_active = True
            self.timer.start(30)  # 30ms = ~33 FPS
            self.update_status("Webcam active - Ready to capture")
        else:
            self.update_status("Error: Could not open webcam", error=True)

    def update_frame(self):
        """Update webcam frame."""
        if not self.webcam_active or self.using_uploaded_image:
            return

        ret, frame = self.webcam.read()
        if ret:
            self.current_frame = frame
            self.display_image(frame, self.main_viewer)

    def display_image(self, image, viewer, max_width=None, max_height=None):
        """Display OpenCV image in viewer (ZoomableImageViewer or QLabel)."""
        if image is None:
            return

        # Convert BGR to RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        h, w = rgb_image.shape[:2]

        # Resize if specified (only for pipeline images)
        if max_width or max_height:
            if max_width and w > max_width:
                scale = max_width / w
                w = max_width
                h = int(h * scale)
            if max_height and h > max_height:
                scale = max_height / h
                h = int(h * scale)
                w = int(w * scale)
            rgb_image = cv2.resize(rgb_image, (w, h))

        # Convert to QImage
        bytes_per_line = 3 * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Display
        pixmap = QPixmap.fromImage(q_image)

        # Check if viewer is ZoomableImageViewer or QLabel
        if isinstance(viewer, ZoomableImageViewer):
            viewer.set_image(pixmap)
        else:
            viewer.setPixmap(pixmap)

    def upload_image(self):
        """Upload an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.current_image = image
                self.using_uploaded_image = True
                self.display_image(image, self.main_viewer)
                self.update_status(f"Loaded: {Path(file_path).name}")
                self.process_btn.setEnabled(True)
                self.clear_pipeline()
            else:
                self.update_status("Error: Could not load image", error=True)

    def capture_image(self):
        """Capture image from webcam or return to live webcam."""
        # If already showing a captured/uploaded image, return to webcam
        if self.using_uploaded_image:
            self.current_image = None
            self.using_uploaded_image = False
            self.feature_combo.setCurrentIndex(0)  # Reset to "None - Live Webcam"
            self.process_btn.setEnabled(False)
            self.clear_pipeline()
            self.update_status("Returned to live webcam")
        # Otherwise, capture the current frame
        elif self.current_frame is not None:
            self.current_image = self.current_frame.copy()
            self.using_uploaded_image = True  # Freeze display
            self.update_status("Image captured from webcam")
            self.process_btn.setEnabled(True)
            self.clear_pipeline()
        else:
            self.update_status("Error: No webcam frame available", error=True)

    def zoom_in(self):
        """Zoom in on main viewer."""
        self.main_viewer.zoom_in()

    def zoom_out(self):
        """Zoom out on main viewer."""
        self.main_viewer.zoom_out()

    def zoom_reset(self):
        """Reset zoom on main viewer."""
        self.main_viewer.reset_zoom()

    def on_replace_toggled(self, state):
        """Handle replacement checkbox toggle."""
        is_checked = state == Qt.CheckState.Checked.value
        self.replace_prompt.setVisible(is_checked)
        self.replace_method_combo.setVisible(is_checked)

    def on_feature_changed(self, feature_name):
        """Handle feature selection change."""
        self.update_status(f"Selected: {feature_name}")

        # Show/hide pipeline viewer and checkboxes based on feature
        if "Will" in feature_name:
            self.pipeline_frame.setVisible(True)
            self.equalize_checkbox.setVisible(True)
            self.replace_checkbox.setVisible(True)
            self.initialize_will_processor()
        else:
            self.pipeline_frame.setVisible(False)
            self.equalize_checkbox.setVisible(False)
            self.replace_checkbox.setVisible(False)
            self.replace_prompt.setVisible(False)
            self.replace_method_combo.setVisible(False)

        # Enable process button if image is loaded
        if self.current_image is not None:
            self.process_btn.setEnabled(True)

        self.clear_pipeline()

    def initialize_will_processor(self):
        """Initialize Will's segmentation processor."""
        if self.processor is not None:
            return  # Already initialized

        try:
            from Will import SegmentationExpanderV2

            # Find SAM model
            assets_dir = Path(__file__).parent / "src" / "Will" / "assets"
            sam_models = list(assets_dir.glob("sam_vit_*.pth"))

            if not sam_models:
                self.update_status("Error: SAM model not found", error=True)
                return

            sam_checkpoint = str(sam_models[0])
            model_type = "vit_b" if "vit_b" in sam_checkpoint.lower() else "vit_h"

            self.update_status("Initializing Will's processor...")
            self.processor = SegmentationExpanderV2(
                sam_checkpoint=sam_checkpoint,
                sam_model_type=model_type,
                yolo_model="yolov8n.pt",
                device="cpu"
            )
            self.update_status("Processor ready!")

        except Exception as e:
            self.update_status(f"Error initializing processor: {e}", error=True)

    def process_image(self):
        """Process the current image with selected feature."""
        if self.current_image is None:
            self.update_status("Error: No image to process", error=True)
            return

        feature_name = self.feature_combo.currentText()

        if feature_name == "None - Live Webcam":
            self.update_status("Please select a feature to process")
            return

        if "Will" in feature_name:
            if self.processor is None:
                self.initialize_will_processor()
                if self.processor is None:
                    return

            # Disable buttons during processing
            self.process_btn.setEnabled(False)
            self.upload_btn.setEnabled(False)
            self.capture_btn.setEnabled(False)

            self.update_status("Processing... Please wait...")

            # Run processing in background thread
            equalize = self.equalize_checkbox.isChecked()
            replace = self.replace_checkbox.isChecked()
            replace_prompt = self.replace_prompt.text().strip() if replace else ""
            replace_method = "controlnet" if "ControlNet" in self.replace_method_combo.currentText() else "sdxl"

            self.processing_thread = ProcessingThread(
                feature_name, self.current_image, self.processor,
                equalize, replace, replace_prompt, replace_method
            )
            self.processing_thread.finished.connect(self.on_processing_finished)
            self.processing_thread.error.connect(self.on_processing_error)
            self.processing_thread.start()

        else:
            self.update_status(f"Feature '{feature_name}' not yet implemented")

    def on_processing_finished(self, results):
        """Handle processing completion."""
        self.processed_results = results

        # Display final result in main viewer
        if 'step5_final' in results:
            self.display_image(results['step5_final'], self.main_viewer)

        # Display pipeline steps
        for step_key, viewer in self.pipeline_viewers.items():
            if step_key in results and results[step_key] is not None:
                self.display_image(results[step_key], viewer, max_height=200)

        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)

        self.update_status("✓ Processing complete!")

    def on_processing_error(self, error_msg):
        """Handle processing error."""
        self.update_status(f"Error during processing: {error_msg}", error=True)

        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)

    def clear_pipeline(self):
        """Clear pipeline viewer."""
        for viewer in self.pipeline_viewers.values():
            viewer.clear()
        self.processed_results = None

    def update_status(self, message, error=False):
        """Update status bar."""
        if error:
            self.status_label.setStyleSheet("padding: 5px; background-color: #e74c3c; color: white;")
        else:
            self.status_label.setStyleSheet("padding: 5px; background-color: #2c3e50; color: white;")
        self.status_label.setText(message)

    def closeEvent(self, event):
        """Handle window close event."""
        if self.webcam is not None:
            self.webcam.release()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = MainGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
