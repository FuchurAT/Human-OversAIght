import os
import sys
from typing import Union, List, Tuple, Optional
from pydantic_core import core_schema
from pydantic import BaseModel, Field
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, QUrl, QRect, QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import threading
import numpy as np
from utils.log import Log
from utils.screen.widgets.loading import Loading
import base64


class ScreenApplication:
    """Base class for screen applications that can dynamically update inputs"""

    def __init__(self, screen):
        self.screen = screen
        self.screen.inputs = []

    def update(self):
        # Update screen if inputs have changed
        if self.input() != self.screen.inputs:
            # Clear only the cells that will be updated
            for idx, _ in enumerate(self.input()):
                row = idx // self.screen.grid_layout[1]
                col = idx % self.screen.grid_layout[1]
                item = self.screen.grid.itemAtPosition(row, col)
                if item and item.widget():
                    item.widget().deleteLater()
            
            self.screen.inputs = self.input()
            self.screen.create_display_widgets()

    def input(self) -> List[str]:
        """Get current inputs to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement input()")


class ScreenConfig(BaseModel):
    """Configuration for a single screen"""

    screen_id: int = Field(..., description="ID of the screen to display on")
    name: str = Field(..., description="Name of the screen")
    inputs: List[str | List] | type[ScreenApplication] = Field(
        ..., description="List of input paths (images/videos) or ScreenApplication class")
    focus_border: str = Field(
        default="green", description="Color of the focus border")
    grid_layout: Optional[tuple[int, int]] = Field(
        default=None, description="Optional fixed grid layout (rows, cols)")
    screen_size: Optional[tuple[int, int]] = Field(
        default=None, description="Optional custom size override (width, height)")

    def validate_inputs(self, value):
        if isinstance(value, list):
            if not all(isinstance(x, str) for x in value):
                raise ValueError(
                    "All inputs must be strings when using a list")
            return value
        elif isinstance(value, type) and issubclass(value, ScreenApplication):
            return value
        else:
            raise ValueError(
                "Inputs must be either a list of strings or a ScreenApplication class")


class ScreenSignals(QObject):
    """Signals for thread-safe communication"""
    show_signal = pyqtSignal()
    close_signal = pyqtSignal()
    create_widgets_signal = pyqtSignal()
    escape_pressed_signal = pyqtSignal()  # New signal for escape key
    # Signal for focus change with screen index
    focus_changed_signal = pyqtSignal(int)


class Screen(QMainWindow):
    def __init__(self, screen_id: int = 0, name: str = "ScreenManager", show_focus: bool = False, screen_size: Optional[tuple[int, int]] = None):
        """
        Initialize the Screen class .

        Args:
            screen_id(int): The ID of the screen to display on(0 for primary screen)
            name(str): Name of the screen for identification
            show_focus(bool): Whether to show focus border
            screen_size(tuple): Optional custom size (width, height)
        """
        # Set window flags before calling parent constructor
        if screen_size:
            super().__init__(None, Qt.Window)
        else:
            super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
            
        self.screen_id = screen_id
        self.name = name
        self.inputs = []
        self.running = False
        self.grid_layout = (1, 1)  # Default single input layout
        self.border_width = 5
        self.focus_border = "blue"
        self.media_players = []
        self.labels = []
        self.spacing = 16
        self.show_focus = show_focus
        self.focused = False
        self.manager = None  # Reference to ScreenManager
        self.screen_size = screen_size  # Initialize screen_size attribute
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self._update_typing)
        self.current_typing_text = ""
        self.current_typing_index = 0
        self.current_typing_label = None
        self.typing_speed = 30  # milliseconds per character

        # Create signals object
        self.signals = ScreenSignals()
        self.signals.show_signal.connect(self._show_window)
        self.signals.close_signal.connect(self._close_window)
        self.signals.create_widgets_signal.connect(
            self._create_display_widgets)
        self.signals.escape_pressed_signal.connect(self._handle_escape)
        self.signals.focus_changed_signal.connect(self._handle_focus_change)

        self.baseStyle = "background: black;"

        if show_focus:
            self.setStyleSheet(f"QMainWindow {{ {self.baseStyle} }}")

        # Initialize UI
        self.init_ui()

        self.messages = []

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"Screen Display - {self.name}")
        
        # Get the target screen
        screens = QApplication.screens()
        if 0 <= self.screen_id < len(screens):
            target_screen = screens[self.screen_id]
            # If screen_size is set, use it to create a custom geometry
            if self.screen_size:
                width, height = self.screen_size
                # Center the window on the target screen
                x = target_screen.geometry().x() + (target_screen.geometry().width() - width) // 2
                y = target_screen.geometry().y() + (target_screen.geometry().height() - height) // 2
                self.setGeometry(x, y, width, height)
            else:
                # Use full screen geometry
                self.setGeometry(target_screen.geometry())
        else:
            Log.warning(
                f"Screen ID {self.screen_id} not found. Using primary screen.")
            if self.screen_size:
                width, height = self.screen_size
                screen = QApplication.primaryScreen().geometry()
                x = screen.x() + (screen.width() - width) // 2
                y = screen.y() + (screen.height() - height) // 2
                self.setGeometry(x, y, width, height)
            else:
                self.setGeometry(QApplication.primaryScreen().geometry())

        # Create central widget for grid layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid = QGridLayout(self.central_widget)
        # self.grid.setSpacing(self.spacing)
        self.grid.setContentsMargins(self.border_width + self.spacing // 2, self.border_width + self.spacing //
                                     2, self.border_width + self.spacing // 2, self.border_width + self.spacing // 2)

    def add_input(self, input_source: Union[List[str], type[ScreenApplication]]):
        """
        Add one or multiple inputs to the screen.

        Args:
            input_source: Can be:
                - List of paths
                - ScreenApplication class
        """
        if isinstance(input_source, list):
            self.inputs.extend(input_source)
        else:
            # Create application instance
            application = input_source(self)
            self.inputs.extend(application.input())

        # Calculate grid layout if not explicitly set
        if self.grid_layout == (1, 1):
            n_inputs = len(self.inputs)
            if n_inputs > 1:
                cols = int(np.ceil(np.sqrt(n_inputs)))
                rows = int(np.ceil(n_inputs / cols))
                self.grid_layout = (rows, cols)

        # Create display widgets
        self.create_display_widgets()

    def _show_window(self):
        """Thread-safe show operation"""
        self.show()

    def _close_window(self):
        """Thread-safe close operation"""
        for player in self.media_players:
            player.stop()
        self.media_players.clear()
        self.labels.clear()
        super().close()

    def _create_display_widgets(self):
        """Thread-safe widget creation"""
        # Clear existing widgets
        for player in self.media_players:
            player.stop()
        self.media_players.clear()
        self.labels.clear()
        
        # Clear the grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get the target screen geometry
        screens = QApplication.screens()
        if 0 <= self.screen_id < len(screens):
            screen = screens[self.screen_id].geometry()
        else:
            screen = QApplication.primaryScreen().geometry()

        # If no inputs, show loading in 1x1 grid
        if not self.inputs:
            loading_widget = Loading()
            self.grid.addWidget(loading_widget, 0, 0)
            return

        # Use configured grid layout for content
        cell_width = screen.width() // self.grid_layout[1] - self.spacing
        cell_height = screen.height() // self.grid_layout[0] - self.spacing

        # Set fixed column widths
        for col in range(self.grid_layout[1]):
            self.grid.setColumnStretch(col, 1)
            self.grid.setColumnMinimumWidth(col, cell_width)

        # Create widgets for each input
        for idx, input in enumerate(self.inputs):
            # Calculate correct row and column
            row = idx // self.grid_layout[1]  # Divide by number of columns to get row
            col = idx % self.grid_layout[1]   # Modulo by number of columns to get column

            # Calculate available space for this cell
            available_width = cell_width - (self.border_width * 2)

            # Check if input is a text string (not a file path)
            if isinstance(input, list):
                # Create a scrollable text area
                text_widget = QWidget()
                text_widget.setStyleSheet("""
                    QWidget {
                        background-color: #2b2b2b;
                        border-radius: 5px;
                    }
                """)
                text_layout = QGridLayout(text_widget)
                text_layout.setContentsMargins(20, 20, 20, 20)

                label = QLabel()
                label.setStyleSheet("""
                    QLabel {
                        color: white;
                        padding: 10px;
                        font-size: 16px;
                        line-height: 24px;
                    }
                """)
                label.setWordWrap(True)
                label.setAlignment(Qt.AlignCenter)  # Center both horizontally and vertically
                
                # Add label to layout
                text_layout.addWidget(label, 0, 0)
                
                # Start typing animation for the message
                self._start_typing_animation(label, "".join(input).replace('"', ''))
                
                self.labels.append(text_widget)
                self.grid.addWidget(text_widget, row, col)

            elif input.lower().endswith(('.mp4', '.avi', '.mov')):
                # Create video widget
                video_widget = QVideoWidget()
                player = QMediaPlayer()
                player.setMuted(True)
                player.setVideoOutput(video_widget)
                player.setMedia(QMediaContent(QUrl.fromLocalFile(input)))

                # Set up looping
                player.mediaStatusChanged.connect(
                    lambda status, p=player: self._handle_media_status(
                        status, p)
                )

                player.play()

                self.media_players.append(player)
                self.grid.addWidget(video_widget, row, col)
            elif input.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Create image label
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)  # Center the image in the container
                
                pixmap = QPixmap(input)
                if pixmap.isNull():
                    Log.warning(f"Failed to load image: {input}")
                    label.setText(f"Error loading:\n{os.path.basename(input)}")
                    label.setStyleSheet("background-color: #FFE4E1; color: red; padding: 10px;")
                    self.grid.addWidget(label, row, col)
                    continue

                # Get original image dimensions
                original_width = pixmap.width()
                original_height = pixmap.height()

                # Calculate scaling factor to fit width while maintaining aspect ratio
                scale_factor = min(1.0, available_width / original_width)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)

                # Scale image to fit available width while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(new_width, new_height,
                                            Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
                
                # Set the scaled pixmap
                label.setPixmap(scaled_pixmap)
                
                # Set size policy to maintain aspect ratio and prevent stretching
                label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                label.setMinimumSize(1, 1)  # Allow shrinking
                label.setMaximumSize(new_width, new_height)  # Prevent stretching beyond original size
                
                # Create a container widget to center the label
                container = QWidget()
                container_layout = QGridLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.addWidget(label, 0, 0, Qt.AlignCenter)
                
                self.grid.addWidget(container, row, col)
            else:
                # Create loading widget
                loading_widget = Loading()
                self.grid.addWidget(loading_widget, row, col)

    def _handle_media_status(self, status, player):
        """Handle media status changes for video looping"""
        if status == QMediaPlayer.EndOfMedia:
            player.setPosition(0)  # Reset to beginning
            player.play()  # Start playing again

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Escape:
            # Emit signal to handle escape in main thread
            self.signals.escape_pressed_signal.emit()
        elif event.key() == Qt.Key_Left:
            # Focus previous screen
            if self.manager:
                self.manager.focus_previous()
        elif event.key() == Qt.Key_Right:
            # Focus next screen
            if self.manager:
                self.manager.focus_next()
        # elif event.key() == Qt.Key_Tab:
        #     self.focusNextChild()
        # super().keyPressEvent(event)

    def focusInEvent(self, event):
        """Handle focus in event"""
        self.focused = True

        if self.show_focus:
            self.setStyleSheet(
                f"QMainWindow {{ {self.baseStyle} border: {self.border_width}px solid {self.focus_border} }}")
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        """Handle focus out event"""
        self.focused = False

        if self.show_focus:
            self.setStyleSheet(f"QMainWindow {{ {self.baseStyle} }}")
        super().focusOutEvent(event)

    def start(self):
        """Start displaying the inputs"""
        if not self.inputs:
            Log.error(f"No input on screen {self.screen_id}")

        self.running = True
        # Use signal to show window from main thread
        self.signals.show_signal.emit()

    def close(self):
        """Close the screen window"""
        self.running = False
        # Use signal to close window from main thread
        self.signals.close_signal.emit()

    def create_display_widgets(self):
        """Create widgets for displaying inputs"""
        # Use signal to create widgets from main thread
        self.signals.create_widgets_signal.emit()

    def _handle_escape(self):
        """Handle escape key press in main thread"""
        if self.manager:
            self.manager.close_all()
            # Force quit the application
            QApplication.quit()
            sys.exit(0)
        else:
            self.close()

    def _handle_focus_change(self, screen_id):
        """Handle focus change signal"""
        if screen_id == self.screen_id:
            self.setFocus()
        else:
            self.clearFocus()

    def _update_typing(self):
        """Update the typing animation"""
        if self.current_typing_label and self.current_typing_index < len(self.current_typing_text):
            current_text = self.current_typing_text[:self.current_typing_index + 1]
            self.current_typing_label.setText(current_text)
            self.current_typing_index += 1
        else:
            self.typing_timer.stop()

    def _start_typing_animation(self, label, text):
        """Start the typing animation for a label"""
        self.current_typing_label = label
        self.current_typing_text = text
        self.current_typing_index = 0
        self.typing_timer.start(self.typing_speed)


class ScreenManager:
    def __init__(self, screens_config: List[ScreenConfig], show_focus: bool = False):
        """
        Initialize the ScreenManager with screen configurations.

        Args:
            screens_config: List of screen configurations
        """
        # Create QApplication instance
        self.app = QApplication(sys.argv)

        self.screens = []
        self.show_focus = show_focus
        self.current_focus = 0

        # Create screens from configurations
        for config in screens_config:
            screen = self.create_screen_from_config(config)
            self.add_screen(screen)

    def add_screen(self, screen: 'Screen'):
        """Add a screen to the manager"""
        self.screens.append(screen)
        screen.manager = self  # Add reference to manager

    def start_all(self):
        """Start all screens in separate threads"""
        for screen in self.screens:
            thread = threading.Thread(target=screen.start)
            thread.daemon = True
            thread.start()

        # Set initial focus
        self._set_focus(self.current_focus)

        # Start the Qt event loop
        self.app.exec_()

    def close_all(self):
        """Close all screens"""
        for screen in self.screens:
            screen.close()
        self.app.quit()

    def focus_next(self):
        """Focus the next screen"""
        if not self.screens:
            return

        # Calculate next screen index
        next_index = (self.current_focus + 1) % len(self.screens)
        self._set_focus(next_index)

    def focus_previous(self):
        """Focus the previous screen"""
        if not self.screens:
            return

        # Calculate previous screen index
        prev_index = (self.current_focus - 1) % len(self.screens)
        self._set_focus(prev_index)

    def _set_focus(self, index):
        """Set focus to a specific screen"""
        if 0 <= index < len(self.screens):
            self.current_focus = index
            # Emit focus change signal to all screens
            for screen in self.screens:
                screen.signals.focus_changed_signal.emit(index)
            # Activate the focused window
            self.screens[index].activateWindow()
            self.screens[index].raise_()

    def create_screen_from_config(self, config: ScreenConfig) -> Screen:
        """Create a screen instance from configuration"""
        screen = Screen(
            screen_id=config.screen_id,
            name=config.name,
            show_focus=self.show_focus,
            screen_size=config.screen_size
        )
        screen.focus_border = config.focus_border
        if config.grid_layout:
            screen.grid_layout = config.grid_layout
        if config.screen_size:
            screen.setFixedSize(*config.screen_size)

        # Create loading widget
        loading_widget = Loading()
        screen.labels.append(loading_widget)
        screen.grid.addWidget(loading_widget, 1, 1)

        screen.add_input(config.inputs)
        return screen
