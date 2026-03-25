import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import os
import chess
import chess.engine
import pandas as pd
import joblib
import json
import threading
from ml_training.feature_extraction import compute_features

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "ml_training", "elo_models")
METRICS_FILE = os.path.join(MODEL_DIR, "model_metrics.json")
FEATURE_SETS_FILE = os.path.join(SCRIPT_DIR, "ml_training", "feature_sets.json")
STOCKFISH_PATH = os.path.join(SCRIPT_DIR, "..", "stockfish-windows-x86-64-avx2.exe")
PIECE_PATH = os.path.join(SCRIPT_DIR, "pieces")  # PNG images: wP.png, bK.png, etc.

# Load metrics and feature sets
with open(METRICS_FILE, "r") as f:
    model_metrics = json.load(f)
with open(FEATURE_SETS_FILE, "r") as f:
    FEATURE_SETS = json.load(f)


def categorize_elo(avg_elo):
    if avg_elo < 800:
        return "800-"
    elif avg_elo <= 1100:
        return "800-1100"
    elif avg_elo <= 1400:
        return "1100-1400"
    elif avg_elo <= 1600:
        return "1400-1600"
    elif avg_elo <= 1800:
        return "1600-1800"
    elif avg_elo <= 2000:
        return "1800-2000"
    elif avg_elo <= 2200:
        return "2000-2200"
    else:
        return "2200+"


# Set modern dark theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChessAnalyserUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("IntuiChess - Advanced Position Analysis")
        self.geometry("1500x800")
        self.minsize(1300, 700)

        # Configure grid weights for responsive layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # Sidebar - fixed width
        self.grid_columnconfigure(1, weight=1)  # Main content - expandable

        # App state
        self.current_page = None
        self.default_elo = 1500  # Default Elo from settings

        # Create main layout
        self.create_sidebar()
        self.create_main_content_area()

        # Initialize with home page
        self.show_page("home")

    def create_sidebar(self):
        """Create modern navigation sidebar"""
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=0, width=200,
                                          fg_color=("#F8F9FA", "#1E1E1E"))
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_propagate(False)

        # Modern app title
        title_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Chess Analyzer",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1F2937", "#F9FAFB")
        )
        title_label.pack(pady=(30, 40))

        # Navigation buttons - modern visible style
        self.nav_buttons = {}

        self.nav_buttons["home"] = ctk.CTkButton(
            self.sidebar_frame,
            text="Board",
            command=lambda: self.show_page("home"),
            width=160,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=("#3B82F6", "#2563EB"),
            hover_color=("#2563EB", "#1D4ED8"),
            corner_radius=8
        )
        self.nav_buttons["home"].pack(pady=(0, 12), padx=20)

        self.nav_buttons["settings"] = ctk.CTkButton(
            self.sidebar_frame,
            text="Settings",
            command=lambda: self.show_page("settings"),
            width=160,
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=("#E5E7EB", "#374151"),
            text_color=("#6B7280", "#9CA3AF"),
            hover_color=("#D1D5DB", "#4B5563"),
            corner_radius=8
        )
        self.nav_buttons["settings"].pack(pady=(0, 12), padx=20)

    def create_main_content_area(self):
        """Create the main content area where pages will be displayed"""
        self.main_content_frame = ctk.CTkFrame(self, corner_radius=15, fg_color=("gray90", "gray13"))
        self.main_content_frame.grid(row=0, column=1, sticky="nsew", padx=(25, 25), pady=25)

        # Configure main content grid
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=1)

    def show_page(self, page_name):
        """Switch to the specified page"""
        # Update navigation button styles - modern approach
        for name, button in self.nav_buttons.items():
            if name == page_name:
                button.configure(
                    fg_color=("#3B82F6", "#2563EB"),
                    text_color="white",
                    hover_color=("#2563EB", "#1D4ED8")
                )
            else:
                button.configure(
                    fg_color=("#E5E7EB", "#374151"),
                    text_color=("#6B7280", "#9CA3AF"),
                    hover_color=("#D1D5DB", "#4B5563")
                )

        # Clear current page
        for widget in self.main_content_frame.winfo_children():
            widget.destroy()

        # Load the requested page
        if page_name == "home":
            self.current_page = HomePage(self.main_content_frame, self)
        elif page_name == "settings":
            self.current_page = SettingsPage(self.main_content_frame, self)

        self.current_page.pack(fill="both", expand=True)


class HomePage(ctk.CTkFrame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        self.board = chess.Board()
        self.selected_square = None
        self.last_move = None
        self.piece_images = {}
        self.move_history = []
        self.redo_stack = []
        self.is_analyzing = False

        self.setup_home_page()

    def setup_home_page(self):
        """Setup modern home page"""
        # Configure grid for home page
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # Board column - fixed width
        self.grid_columnconfigure(1, weight=1)  # Control column - expandable

        # Left: Board Section - Modern card
        self.board_frame = ctk.CTkFrame(self, corner_radius=12,
                                        fg_color=("white", "#2D2D2D"))
        self.board_frame.grid(row=0, column=0, sticky="nsew", padx=(25, 15), pady=25)

        # Board title
        board_title = ctk.CTkLabel(self.board_frame, text="Position Board",
                                   font=ctk.CTkFont(size=18, weight="bold"),
                                   text_color=("#1F2937", "#F9FAFB"))
        board_title.pack(pady=(20, 15))

        # Modern board container
        self.board_container = ctk.CTkFrame(self.board_frame, corner_radius=8,
                                            fg_color=("#F8F9FA", "#1A1A1A"))
        self.board_container.pack(padx=20, pady=(0, 15))

        self.board_canvas = tk.Canvas(self.board_container, width=520, height=520,
                                      bg="#F8F9FA", highlightthickness=0, relief='flat')
        self.board_canvas.pack(padx=12, pady=12)
        self.board_canvas.bind("<Button-1>", self.on_square_click)

        # Modern control buttons
        self.control_buttons_frame = ctk.CTkFrame(self.board_frame, fg_color="transparent")
        self.control_buttons_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.undo_button = ctk.CTkButton(
            self.control_buttons_frame, text="← Undo", height=40,
            command=self.undo_move, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#F3F4F6", "#4B5563"), text_color=("#374151", "#D1D5DB"),
            hover_color=("#E5E7EB", "#6B7280"),
            corner_radius=8
        )
        self.undo_button.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.redo_button = ctk.CTkButton(
            self.control_buttons_frame, text="Redo →", height=40,
            command=self.redo_move, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#F3F4F6", "#4B5563"), text_color=("#374151", "#D1D5DB"),
            hover_color=("#E5E7EB", "#6B7280"),
            corner_radius=8
        )
        self.redo_button.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # Right: Modern Analysis Panel
        self.control_frame = ctk.CTkFrame(self, corner_radius=12,
                                          fg_color=("white", "#2D2D2D"))
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=(15, 25), pady=25)

        # Configure control frame grid
        self.control_frame.grid_rowconfigure(3, weight=1)  # Results section expands
        self.control_frame.grid_columnconfigure(0, weight=1)

        # Panel title
        control_title = ctk.CTkLabel(self.control_frame, text="Analysis Controls",
                                     font=ctk.CTkFont(size=18, weight="bold"),
                                     text_color=("#1F2937", "#F9FAFB"))
        control_title.grid(row=0, column=0, pady=(25, 20))

        # Setup Card
        setup_card = ctk.CTkFrame(self.control_frame, corner_radius=8,
                                  fg_color=("#F8F9FA", "#374151"))
        setup_card.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 15))

        # FEN Input
        fen_label = ctk.CTkLabel(setup_card, text="Position (FEN)",
                                 font=ctk.CTkFont(size=13, weight="bold"),
                                 text_color=("#374151", "#D1D5DB"))
        fen_label.pack(anchor="w", padx=15, pady=(15, 5))

        self.fen_entry = ctk.CTkEntry(setup_card, height=40,
                                      font=ctk.CTkFont(size=12),
                                      placeholder_text="Enter FEN string...",
                                      corner_radius=6,
                                      fg_color=("white", "#4B5563"),
                                      border_width=1,
                                      border_color=("#D1D5DB", "#6B7280"))
        self.fen_entry.pack(fill="x", padx=15, pady=(0, 10))
        self.fen_entry.insert(0, chess.STARTING_FEN)

        # Time control
        time_label = ctk.CTkLabel(setup_card, text="Time Control",
                                  font=ctk.CTkFont(size=13, weight="bold"),
                                  text_color=("#374151", "#D1D5DB"))
        time_label.pack(anchor="w", padx=15, pady=(5, 5))

        self.time_var = tk.StringVar(value="blitz")
        time_menu = ctk.CTkOptionMenu(setup_card, variable=self.time_var,
                                      values=["blitz", "rapid_classical"],
                                      height=40, font=ctk.CTkFont(size=12),
                                      corner_radius=6,
                                      fg_color=("#3B82F6", "#2563EB"),
                                      button_color=("#2563EB", "#1D4ED8"),
                                      button_hover_color=("#1D4ED8", "#1E40AF"))
        time_menu.pack(fill="x", padx=15, pady=(0, 15))

        # Action buttons card
        button_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        button_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 15))
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        self.analyse_button = ctk.CTkButton(button_frame, text="🔍 Analyze",
                                            command=self.load_fen, height=44,
                                            font=ctk.CTkFont(size=14, weight="bold"),
                                            fg_color=("#10B981", "#059669"),
                                            hover_color=("#059669", "#047857"),
                                            corner_radius=8)
        self.analyse_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.reset_button = ctk.CTkButton(button_frame, text="🔄 Reset",
                                          command=self.reset_board, height=44,
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          fg_color=("#EF4444", "#DC2626"),
                                          hover_color=("#DC2626", "#B91C1C"),
                                          corner_radius=8)
        self.reset_button.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # Results card
        results_card = ctk.CTkFrame(self.control_frame, corner_radius=8,
                                    fg_color=("#F8F9FA", "#374151"))
        results_card.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 15))
        results_card.grid_rowconfigure(1, weight=1)
        results_card.grid_columnconfigure(0, weight=1)

        results_label = ctk.CTkLabel(results_card, text="Analysis Results",
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     text_color=("#374151", "#D1D5DB"))
        results_label.grid(row=0, column=0, pady=(15, 10), padx=15, sticky="w")

        self.result_text = ctk.CTkTextbox(results_card,
                                          font=ctk.CTkFont(size=12),
                                          corner_radius=6,
                                          fg_color=("white", "#4B5563"),
                                          border_width=1,
                                          border_color=("#D1D5DB", "#6B7280"))
        self.result_text.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Status label
        self.status_label = ctk.CTkLabel(self.control_frame, text="",
                                         font=ctk.CTkFont(size=11),
                                         text_color=("#6B7280", "#9CA3AF"))
        self.status_label.grid(row=4, column=0, sticky="w", padx=20, pady=(0, 15))

        # Add minimal welcome text
        welcome_text = (
            "Chess position analysis\n\n"
            "Click pieces to move\n"
            "Enter FEN or use board\n"
            "Click Analyze for insights"
        )
        self.result_text.insert("0.0", welcome_text)

        self.draw_board()

    def update_status(self, message, status_type="info"):
        """Update status indicator with colored feedback"""
        colors = {
            "info": ("gray60", "gray40"),
            "loading": ("#3B82F6", "#1D4ED8"),
            "success": ("#10B981", "#047857"),
            "error": ("#EF4444", "#DC2626")
        }
        self.status_label.configure(text=message, text_color=colors.get(status_type, colors["info"]))
        self.update_idletasks()  # Force UI update

    # --- Board functions ---
    def reset_board(self):
        self.board.reset()
        self.selected_square = None
        self.last_move = None
        self.move_history.clear()
        self.redo_stack.clear()
        self.is_analyzing = False
        self.draw_board()
        self.result_text.delete("0.0", tk.END)
        self.fen_entry.delete(0, tk.END)
        self.fen_entry.insert(0, chess.STARTING_FEN)
        self.analyse_button.configure(state="normal", text="📊 Analyze Position")
        self.update_status("Board reset to starting position", "success")

    def load_fen(self):
        if self.is_analyzing:
            self.update_status("Analysis already in progress...", "loading")
            return

        fen = self.fen_entry.get().strip()
        if not fen:
            self.update_status("Please enter a valid FEN position", "error")
            return

        try:
            self.board.set_fen(fen)
            self.update_status("FEN loaded successfully", "success")
        except Exception as e:
            self.result_text.delete("0.0", tk.END)
            error_msg = f"\u274c Invalid FEN Position\n\nError: {str(e)}\n\nPlease check your FEN string and try again."
            self.result_text.insert(tk.END, error_msg)
            self.update_status(f"Error: {str(e)}", "error")
            return

        self.selected_square = None
        self.last_move = None
        self.move_history.clear()
        self.redo_stack.clear()
        self.draw_board()

        # Disable analyze button during analysis
        self.analyse_button.configure(state="disabled", text="🔄 Analyzing...")
        threading.Thread(target=self.run_analysis, daemon=True).start()

    # --- Click-to-move ---
    def on_square_click(self, event):
        size = 60
        offset = 20  # Account for coordinate labels
        col = (event.x - offset) // size
        row = (event.y - offset) // size

        # Ensure clicks are within board bounds
        if col < 0 or col > 7 or row < 0 or row > 7:
            return
        square = chess.square(col, 7 - row)
        piece = self.board.piece_at(square)

        if self.selected_square is None:
            # Select a piece to move
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.draw_board()  # redraw to show highlight
        else:
            # Check if clicking on the same square to deselect
            if square == self.selected_square:
                self.selected_square = None
                self.draw_board()
                return

            # Check if clicking on another piece of the same color to switch selection
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.draw_board()
                return

            # Try moving the selected piece
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move)
                self.redo_stack.clear()
                self.last_move = move
                # Clear selection only after move
                self.selected_square = None
                self.draw_board()
                self.update_status(f"Move played: {move}", "info")
                self.analyse_button.configure(state="disabled", text="🔄 Analyzing...")
                threading.Thread(target=self.run_analysis, daemon=True).start()
            else:
                # If invalid move, deselect the piece
                self.selected_square = None
                self.draw_board()

    # --- Undo / Redo ---
    def undo_move(self):
        if self.is_analyzing:
            self.update_status("Please wait for current analysis to complete", "loading")
            return

        if self.move_history:
            move = self.move_history.pop()
            self.board.pop()
            self.redo_stack.append(move)
            self.last_move = self.move_history[-1] if self.move_history else None
            self.draw_board()
            self.update_status(f"Undid move: {move}", "info")
            self.analyse_button.configure(state="disabled", text="🔄 Analyzing...")
            threading.Thread(target=self.run_analysis, daemon=True).start()
        else:
            self.update_status("No moves to undo", "info")

    def redo_move(self):
        if self.is_analyzing:
            self.update_status("Please wait for current analysis to complete", "loading")
            return

        if self.redo_stack:
            move = self.redo_stack.pop()
            self.board.push(move)
            self.move_history.append(move)
            self.last_move = move
            self.draw_board()
            self.update_status(f"Redid move: {move}", "info")
            self.analyse_button.configure(state="disabled", text="🔄 Analyzing...")
            threading.Thread(target=self.run_analysis, daemon=True).start()
        else:
            self.update_status("No moves to redo", "info")

    # --- Draw Board ---
    def draw_board(self):
        self.board_canvas.delete("all")
        size = 60
        offset = 20  # Space for coordinates

        # Draw coordinate labels - minimal style
        # Files (a-h) at bottom
        for i, file_char in enumerate('abcdefgh'):
            self.board_canvas.create_text(
                offset + i * size + size // 2, 500,
                text=file_char, fill="#9CA3AF",
                font=("Segoe UI", 11, "normal")
            )

        # Ranks (1-8) on left
        for i, rank_num in enumerate('87654321'):
            self.board_canvas.create_text(
                10, offset + i * size + size // 2,
                text=rank_num, fill="#9CA3AF",
                font=("Segoe UI", 11, "normal")
            )

        # Draw board squares
        for r in range(8):
            for c in range(8):
                square = chess.square(c, 7 - r)
                x1, y1 = offset + c * size, offset + r * size
                x2, y2 = x1 + size, y1 + size

                # Base colors - modern but visible
                if (r + c) % 2 == 0:
                    light_color = "#F0F9FF"  # Light squares - subtle blue tint
                    selected_light = "#DBEAFE"  # Selected light - blue
                else:
                    light_color = "#E0E7FF"  # Dark squares - light indigo
                    selected_light = "#C7D2FE"  # Selected dark - indigo

                base_color = light_color

                # Highlight last move - visible but modern
                if self.last_move and (square == self.last_move.from_square or square == self.last_move.to_square):
                    base_color = "#FEF08A" if (r + c) % 2 == 0 else "#FDE047"

                # Highlight selected piece
                if self.selected_square == square:
                    base_color = selected_light

                # Draw square without border for minimal look
                self.board_canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=base_color, outline=base_color, width=0
                )

                # Add legal move indicators for selected piece
                if self.selected_square is not None:
                    for move in self.board.legal_moves:
                        if move.from_square == self.selected_square and move.to_square == square:
                            # Draw modern legal move indicators
                            center_x, center_y = x1 + size // 2, y1 + size // 2
                            radius = 8
                            self.board_canvas.create_oval(
                                center_x - radius, center_y - radius,
                                center_x + radius, center_y + radius,
                                fill="#3B82F6", outline="#2563EB", width=2
                            )

        # Draw pieces
        for square, piece in self.board.piece_map().items():
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            x = offset + col * size
            y = offset + row * size

            filename = f"{piece.color and 'w' or 'b'}{piece.symbol().upper()}.png"
            if filename not in self.piece_images:
                img_path = os.path.join(PIECE_PATH, filename)
                if os.path.exists(img_path):
                    self.piece_images[filename] = ImageTk.PhotoImage(
                        Image.open(img_path).resize((size, size), Image.Resampling.LANCZOS)
                    )
                else:
                    # Fallback to text if image not found
                    piece_text = piece.symbol().upper() if piece.color else piece.symbol().lower()
                    self.board_canvas.create_text(
                        x + size // 2, y + size // 2,
                        text=piece_text, fill="black" if piece.color else "white",
                        font=("Arial", 24, "bold")
                    )
                    continue

            self.board_canvas.create_image(x, y, anchor="nw", image=self.piece_images[filename])

    # --- Analysis ---
    def run_analysis(self):
        self.is_analyzing = True
        self.update_status("🔍 Starting analysis...", "loading")

        try:
            avg_elo = self.main_app.default_elo  # Use default from settings
            time_control = self.time_var.get()
            elo_range = categorize_elo(avg_elo)

            self.update_status("⚙️ Computing position features...", "loading")

            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
                features = compute_features(self.board, engine)

            self.update_status("🧠 Running ML predictions...", "loading")

            targets = [("label_position_quality", "Position Quality"),
                       ("label_move_ease", "Move Ease")]
            results = []
            header = f"Analysis\n\n"
            results.append(header)

            analysis_found = False
            for target, label_name in targets:
                feature_cols = FEATURE_SETS.get(elo_range, {}).get(target, FEATURE_SETS["default"][target])
                X = pd.DataFrame([{k: features[k] for k in feature_cols}])
                model_path = os.path.join(MODEL_DIR, f"model_{elo_range}_{time_control}_{target}.pkl")

                if not os.path.exists(model_path):
                    results.append(f"{label_name}: No model available\n")
                    continue

                analysis_found = True
                model = joblib.load(model_path)
                predicted_score = model.predict(X)[0]

                metrics = model_metrics.get(elo_range, {}).get(time_control, {}).get(target, {})
                rmse = metrics.get("rmse", None)
                certainty = max(0.0, 1.0 - rmse) if rmse else 0.0

                result_line = f"{label_name}: {predicted_score:.3f}\n"
                results.append(result_line)

            if analysis_found:
                self.update_status("Analysis complete", "success")
            else:
                self.update_status("No models available", "error")
                results.append("No models found for this configuration\n")

        except ValueError as e:
            error_msg = "Invalid Elo rating"
            results = [error_msg]
            self.update_status("Invalid input", "error")

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            results = [error_msg]
            self.update_status("Analysis failed", "error")

        finally:
            self.is_analyzing = False
            # Re-enable analyze button
            self.analyse_button.configure(state="normal", text="📊 Analyze Position")

            # Update results display
            self.result_text.delete("0.0", tk.END)
            self.result_text.insert(tk.END, "".join(results))


class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app

        self.setup_settings_page()

    def setup_settings_page(self):
        """Setup the settings page layout"""
        # Configure grid for settings page
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Settings title
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="ew", padx=40, pady=(40, 20))

        settings_title = ctk.CTkLabel(
            title_frame,
            text="⚙️ Application Settings",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        settings_title.pack(pady=(0, 10))

        subtitle = ctk.CTkLabel(
            title_frame,
            text="Configure your Chess Analyzer preferences",
            font=ctk.CTkFont(size=16),
            text_color=("gray60", "gray40")
        )
        subtitle.pack()

        # Main settings container
        settings_container = ctk.CTkScrollableFrame(self, corner_radius=15)
        settings_container.grid(row=1, column=0, sticky="nsew", padx=40, pady=(0, 40))
        settings_container.grid_columnconfigure(0, weight=1)

        # Player Settings Section
        self.create_player_settings(settings_container)

        # Appearance Settings Section
        self.create_appearance_settings(settings_container)

        # About Section
        self.create_about_section(settings_container)

    def create_player_settings(self, parent):
        """Create player settings section"""
        player_section = ctk.CTkFrame(parent, corner_radius=12)
        player_section.grid(row=0, column=0, sticky="ew", pady=(0, 20), padx=20)
        player_section.grid_columnconfigure(1, weight=1)

        # Section header
        header_frame = ctk.CTkFrame(player_section, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=25, pady=(25, 15))

        section_title = ctk.CTkLabel(
            header_frame,
            text="🎯 Player Configuration",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        section_title.pack(anchor="w")

        section_desc = ctk.CTkLabel(
            header_frame,
            text="Set default player parameters for analysis",
            font=ctk.CTkFont(size=14),
            text_color=("gray60", "gray40")
        )
        section_desc.pack(anchor="w", pady=(5, 0))

        # Default Elo Setting
        elo_label = ctk.CTkLabel(
            player_section,
            text="Default Player Elo:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        elo_label.grid(row=1, column=0, sticky="w", padx=25, pady=(0, 10))

        elo_frame = ctk.CTkFrame(player_section, fg_color="transparent")
        elo_frame.grid(row=1, column=1, sticky="ew", padx=25, pady=(0, 10))

        self.elo_var = tk.StringVar(value=str(self.main_app.default_elo))
        self.elo_entry = ctk.CTkEntry(
            elo_frame,
            textvariable=self.elo_var,
            width=150,
            height=35,
            font=ctk.CTkFont(size=14),
            placeholder_text="e.g., 1500"
        )
        self.elo_entry.pack(side="left")

        elo_info = ctk.CTkLabel(
            elo_frame,
            text="(800-3000 range)",
            font=ctk.CTkFont(size=12),
            text_color=("gray60", "gray40")
        )
        elo_info.pack(side="left", padx=(10, 0))

        # Elo description
        elo_desc = ctk.CTkLabel(
            player_section,
            text="This will be used as the default Elo rating for new analysis sessions.",
            font=ctk.CTkFont(size=12),
            text_color=("gray60", "gray40")
        )
        elo_desc.grid(row=2, column=0, columnspan=2, sticky="w", padx=25, pady=(0, 25))

    def create_appearance_settings(self, parent):
        """Create appearance settings section"""
        appearance_section = ctk.CTkFrame(parent, corner_radius=12)
        appearance_section.grid(row=1, column=0, sticky="ew", pady=(0, 20), padx=20)
        appearance_section.grid_columnconfigure(1, weight=1)

        # Section header
        header_frame = ctk.CTkFrame(appearance_section, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=25, pady=(25, 15))

        section_title = ctk.CTkLabel(
            header_frame,
            text="🎨 Appearance",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        section_title.pack(anchor="w")

        section_desc = ctk.CTkLabel(
            header_frame,
            text="Customize the look and feel of the application",
            font=ctk.CTkFont(size=14),
            text_color=("gray60", "gray40")
        )
        section_desc.pack(anchor="w", pady=(5, 0))

        # Theme Setting
        theme_label = ctk.CTkLabel(
            appearance_section,
            text="Theme Mode:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        theme_label.grid(row=1, column=0, sticky="w", padx=25, pady=(0, 10))

        theme_frame = ctk.CTkFrame(appearance_section, fg_color="transparent")
        theme_frame.grid(row=1, column=1, sticky="ew", padx=25, pady=(0, 10))

        self.theme_var = tk.StringVar(value=ctk.get_appearance_mode().lower())
        theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            variable=self.theme_var,
            values=["light", "dark", "system"],
            width=150,
            height=35,
            font=ctk.CTkFont(size=14),
            command=self.change_theme
        )
        theme_menu.pack(side="left")

        # Theme description
        theme_desc = ctk.CTkLabel(
            appearance_section,
            text="Choose between light, dark, or system theme mode.",
            font=ctk.CTkFont(size=12),
            text_color=("gray60", "gray40")
        )
        theme_desc.grid(row=2, column=0, columnspan=2, sticky="w", padx=25, pady=(0, 25))

        # Preview section
        preview_frame = ctk.CTkFrame(appearance_section)
        preview_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=25, pady=(0, 25))

        preview_title = ctk.CTkLabel(
            preview_frame,
            text="🔍 Theme Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        preview_title.pack(pady=(15, 10))

        # Sample UI elements
        sample_frame = ctk.CTkFrame(preview_frame)
        sample_frame.pack(padx=20, pady=(0, 15))

        sample_button = ctk.CTkButton(
            sample_frame,
            text="Sample Button",
            width=120,
            height=32
        )
        sample_button.pack(side="left", padx=10, pady=10)

        sample_entry = ctk.CTkEntry(
            sample_frame,
            placeholder_text="Sample Input",
            width=120,
            height=32
        )
        sample_entry.pack(side="left", padx=10, pady=10)

    def create_about_section(self, parent):
        """Create about section"""
        about_section = ctk.CTkFrame(parent, corner_radius=12)
        about_section.grid(row=2, column=0, sticky="ew", pady=(0, 20), padx=20)

        # Section header
        header_frame = ctk.CTkFrame(about_section, fg_color="transparent")
        header_frame.pack(fill="x", padx=25, pady=(25, 15))

        section_title = ctk.CTkLabel(
            header_frame,
            text="ℹ️ About Chess Analyzer",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        section_title.pack(anchor="w")

        # About content
        about_text = (
            "Chess Analyzer is an advanced position analysis tool powered by machine learning.\n\n"
            "• Real-time position evaluation\n"
            "• Move ease assessment\n"
            "• Player-level specific predictions\n"
            "• Interactive chess board interface\n\n"
            "Built with Python, CustomTkinter, and Python-Chess."
        )

        about_label = ctk.CTkLabel(
            about_section,
            text=about_text,
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        about_label.pack(anchor="w", padx=25, pady=(0, 20))

        # Action buttons
        button_frame = ctk.CTkFrame(about_section, fg_color="transparent")
        button_frame.pack(fill="x", padx=25, pady=(0, 25))

        save_button = ctk.CTkButton(
            button_frame,
            text="💾 Save Settings",
            command=self.save_settings,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#10B981", "#047857"),
            hover_color=("#059669", "#065F46")
        )
        save_button.pack(side="left", padx=(0, 10))

        reset_button = ctk.CTkButton(
            button_frame,
            text="🔄 Reset to Defaults",
            command=self.reset_settings,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#EF4444", "#DC2626"),
            hover_color=("#DC2626", "#B91C1C")
        )
        reset_button.pack(side="left")

        self.status_label = ctk.CTkLabel(
            button_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="right", padx=(20, 0))

    def change_theme(self, theme_choice):
        """Change the application theme"""
        ctk.set_appearance_mode(theme_choice)
        self.update_status("✨ Theme changed successfully!", "success")

    def save_settings(self):
        """Save current settings"""
        try:
            # Validate and save Elo
            new_elo = int(self.elo_var.get())
            if 800 <= new_elo <= 3000:
                self.main_app.default_elo = new_elo
                self.update_status("✅ Settings saved successfully!", "success")
            else:
                self.update_status("⚠️ Elo must be between 800-3000", "error")
        except ValueError:
            self.update_status("❌ Invalid Elo value. Please enter a number.", "error")

    def reset_settings(self):
        """Reset settings to defaults"""
        self.main_app.default_elo = 1500
        self.elo_var.set("1500")
        self.theme_var.set("dark")
        ctk.set_appearance_mode("dark")
        self.update_status("🔄 Settings reset to defaults", "success")

    def update_status(self, message, status_type="info"):
        """Update status message with color coding"""
        colors = {
            "success": ("#10B981", "#047857"),
            "error": ("#EF4444", "#DC2626"),
            "info": ("gray60", "gray40")
        }
        self.status_label.configure(
            text=message,
            text_color=colors.get(status_type, colors["info"])
        )
        # Auto-clear status after 3 seconds
        self.after(3000, lambda: self.status_label.configure(text=""))


if __name__ == "__main__":
    app = ChessAnalyserUI()
    app.mainloop()
