from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QFontMetrics
import math

class Loading(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0  # Use private variable for angle
        self.nodes = 8  # Reduced number of nodes for inner network
        self.node_size = 12  # Larger nodes
        self.spacing = 45  # Increased spacing for larger network
        self.text_spacing = 30  # Spacing between network and text
        self.color = QColor(0, 255, 255)  # Cyan color
        self.setMinimumSize(200, 200)  # Larger minimum size
        
        # Create rotation animation
        self.animation = QPropertyAnimation(self, b"angle")
        self.animation.setDuration(3000)  # 3 seconds per rotation
        self.animation.setStartValue(0)
        self.animation.setEndValue(360)
        self.animation.setEasingCurve(QEasingCurve.Linear)
        self.animation.setLoopCount(-1)
        self.animation.start()

        # Create dots animation
        self._dots = 0
        self.dots_animation = QPropertyAnimation(self, b"dots")
        self.dots_animation.setDuration(2000)
        self.dots_animation.setStartValue(0)
        self.dots_animation.setEndValue(4)
        self.dots_animation.setLoopCount(-1)
        self.dots_animation.setEasingCurve(QEasingCurve.Linear)
        self.dots_animation.start()

        # Calculate character width for monospace font
        self.font = QFont("Consolas", 12, QFont.Bold)  # Use monospace font
        self.char_width = 0
        self.text_height = 0
        self.update_font_metrics()

    def update_font_metrics(self):
        """Update font metrics when font changes"""
        metrics = QFontMetrics(self.font)
        self.char_width = metrics.width("X")  # Use 'X' as reference character
        self.text_height = metrics.height()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate center
        center_x = self.width() / 2
        center_y = self.height() / 2 - self.text_spacing / 2  # Move network up to make room for text
        
        # Draw white hemisphere circle
        painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
        painter.setBrush(Qt.NoBrush)
        hemisphere_radius = self.spacing + 10
        painter.drawEllipse(int(center_x - hemisphere_radius),
                          int(center_y - hemisphere_radius),
                          int(hemisphere_radius * 2 - 2),
                          int(hemisphere_radius * 2 - 2))
        
        # Draw nodes and connections
        node_positions = []
        for i in range(self.nodes):
            # Calculate node position
            angle = self._angle + (i * (360 / self.nodes))
            rad_angle = math.radians(angle)
            
            x = center_x + math.cos(rad_angle) * self.spacing
            y = center_y + math.sin(rad_angle) * self.spacing
            node_positions.append((x, y))
            
            # Draw connections
            for j in range(i + 1, self.nodes):
                # Calculate connected node position
                conn_angle = self._angle + (j * (360 / self.nodes))
                conn_rad_angle = math.radians(conn_angle)
                
                conn_x = center_x + math.cos(conn_rad_angle) * self.spacing
                conn_y = center_y + math.sin(conn_rad_angle) * self.spacing
                
                # Draw constant connection
                painter.setPen(QPen(QColor(0, 255, 255, 100), 1))
                painter.drawLine(int(x), int(y), int(conn_x), int(conn_y))
            
            # Draw node with constant glow
            # Inner glow
            gradient = QLinearGradient(int(x - self.node_size), int(y - self.node_size),
                                     int(x + self.node_size), int(y + self.node_size))
            gradient.setColorAt(0, QColor(0, 255, 255, 200))
            gradient.setColorAt(1, QColor(0, 255, 255, 0))

        # Draw "I'm thinking" text with animated dots
        painter.setFont(self.font)
        base_text = "I'm thinking"
        dots = "." * int(self._dots)
        
        # Calculate total width and position
        total_width = len(base_text + "...") * self.char_width
        text_x = center_x - total_width / 2
        text_y = center_y + self.spacing + self.text_spacing + 15
        
        # Draw text with glow
        for i in range(3):
            glow_color = QColor(0, 255, 255, int(100 * (1 - i/3) * 1))
            painter.setPen(QPen(glow_color, 2))
            
            # Draw base text
            painter.drawText(
                int(text_x),
                int(text_y),
                base_text
            )
            
            # Draw dots
            dots_x = text_x + len(base_text) * self.char_width
            painter.drawText(
                int(dots_x),
                int(text_y),
                dots
            )

    @pyqtProperty(float)
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self.update()

    @pyqtProperty(float)
    def dots(self):
        return self._dots

    @dots.setter
    def dots(self, value):
        self._dots = value
        self.update()