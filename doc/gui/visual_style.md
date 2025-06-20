Break down how to achieve the visual style using Tkinter. We'll need to use a combination of standard widgets, styling, potentially a `Canvas` for the waveform, and images for the icons/buttons to get the closest look.

**Key Libraries:**

1.  **Tkinter:** For the basic window structure, frames, labels, and canvas.
2.  **Tkinter TTK:** For potentially better-looking themed widgets (optional, but often preferred).
3.  **Pillow (PIL):** Essential for loading and using custom images (like `.png`) for the buttons, as Tkinter's built-in support is limited (GIF/PGM). Install with `pip install Pillow`.

**Component Breakdown and Tkinter Implementation:**

1.  **Main Window (Outer Container):**
    *   **Visual:** Light green background, rounded corners, thick black border, 'X' close button.
    *   **Tkinter:**
        *   `root = TkinterDnD.Tk()` (or `tk.Tk()` if not using DnD yet)
        *   `root.config(bg="#C8E6C9")` (Approx light green)
        *   **Rounded Corners:** Very difficult/OS-dependent for the *main* window in Tkinter. Often faked by creating a main `Frame` inside the window, giving *that* the background color and border, and leaving the actual window corners square. Let's start with square corners for the main window.
        *   **Border:** You can't easily add a border to the root window itself. Again, use an inner main `Frame` with `borderwidth=2, relief=tk.SOLID`.
        *   **'X' Button:** Handled by the OS window manager.

2.  **Display Area:**
    *   **Visual:** Inner rounded rectangle, pinkish background, contains waveform, darker border.
    *   **Tkinter:**
        *   Use a `tk.Canvas` widget. It allows drawing custom shapes (waveform).
        *   `display_canvas = tk.Canvas(root, bg="#FFCDD2", height=100, borderwidth=2, relief=tk.SUNKEN)` (Approx pink, adjust height). Add `highlightthickness=0` to remove the default focus border if needed.
        *   Pack this canvas into the main frame/window.

3.  **Status Text ("Thinking..."):**
    *   **Visual:** Simple text label below the display.
    *   **Tkinter:**
        *   `status_label = ttk.Label(root, text="Initializing...", font=("Helvetica", 10))` (or `tk.Label`)
        *   Pack it below the display canvas. Use `anchor=tk.W` (west) or `tk.CENTER` for alignment.

4.  **Button Area (Frame):**
    *   **Visual:** Holds the three controls horizontally.
    *   **Tkinter:**
        *   `button_frame = ttk.Frame(root)` (or `tk.Frame`)
        *   Pack it below the status label. Set its background to match the main green if needed: `button_frame.config(style="Green.TFrame")` after defining the style.

5.  **Context Button/Indicator:**
    *   **Visual:** Rectangle, black border, green fill when ON, small inner green bar. Clickable toggle.
    *   **Tkinter (Best Approach: Image Button):**
        *   Create two images externally (e.g., `context_on.png`, `context_off.png`) that look exactly like the button in both states. Make sure they have transparency if needed.
        *   Load images using Pillow:
            ```python
            from PIL import Image, ImageTk
            # Load OFF state
            img_off_pil = Image.open("context_off.png").resize((width, height)) # Adjust path/size
            self.context_img_off = ImageTk.PhotoImage(img_off_pil)
            # Load ON state
            img_on_pil = Image.open("context_on.png").resize((width, height))
            self.context_img_on = ImageTk.PhotoImage(img_on_pil)
            ```
        *   Create a `tk.Button` (using `tk.Button` often works better for images than `ttk.Button`):
            ```python
            self.context_button = tk.Button(button_frame, image=self.context_img_off,
                                            command=self.toggle_context,
                                            borderwidth=0, relief=tk.FLAT, activebackground="#C8E6C9") # Match bg
            ```
        *   In `toggle_context` and `_load_context_thread`, change the button's image: `self.context_button.config(image=self.context_img_on)` or `self.context_img_off`.

6.  **Download Button:**
    *   **Visual:** Square-ish button with download icon.
    *   **Tkinter (Image Button):**
        *   Create or find a download icon (`download_icon.png`).
        *   Load with Pillow:
            ```python
            img_dl_pil = Image.open("download_icon.png").resize((icon_size, icon_size))
            self.download_icon = ImageTk.PhotoImage(img_dl_pil)
            # Also create a disabled version (e.g., greyscale) if desired
            img_dl_disabled_pil = Image.open("download_icon_disabled.png").resize(...)
            self.download_icon_disabled = ImageTk.PhotoImage(img_dl_disabled_pil)
            ```
        *   Create the `tk.Button`:
            ```python
            self.download_button = tk.Button(button_frame, image=self.download_icon_disabled, # Start disabled
                                             command=self.download_audio, state=tk.DISABLED,
                                             borderwidth=0, relief=tk.FLAT, ...)
            ```
        *   Update the `image` and `state` when enabling/disabling.

7.  **Record Button:**
    *   **Visual:** Red circular button.
    *   **Tkinter (Image Button):**
        *   Create a red circle image (`record_button.png`), maybe one slightly different for a pressed state (`record_button_active.png`) or disabled state (`record_button_disabled.png`).
        *   Load with Pillow similarly.
        *   Create `tk.Button`:
            ```python
            self.record_button = tk.Button(button_frame, image=self.record_icon_disabled, # Start disabled
                                           command=self.handle_read_context_request, state=tk.DISABLED,
                                           borderwidth=0, relief=tk.FLAT, ...)
            ```
        *   Update `image` and `state` based on context loaded / processing state.

**Putting it Together (Conceptual Code Snippet):**

```python
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time # For simulation
import os
import shutil # For file copying
from tkinterdnd2 import DND_FILES, TkinterDnD # If using DnD
from PIL import Image, ImageTk # Import Pillow
import random # For waveform simulation

# Define Colors
BG_GREEN = "#C8E6C9"
DISPLAY_PINK = "#FFCDD2"
WAVE_RED = "#A00000"
BORDER_COLOR = "#000000" # Black

class VoiceApp(TkinterDnD.Tk): # Or tk.Tk if no DnD
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.title("Voice Assistant")
        self.geometry("450x300") # Adjusted size
        self.config(bg=BG_GREEN)
        # Prevent resizing?
        # self.resizable(False, False)

        # --- Load Images ---
        self.load_images() # Call a method to load all images

        # --- Main Frame (for border/padding) ---
        # Use a frame to easily manage padding and maybe a border
        main_frame = tk.Frame(self, bg=BG_GREEN, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Display Canvas ---
        self.display_canvas = tk.Canvas(main_frame, bg=DISPLAY_PINK, height=100,
                                        borderwidth=2, relief=tk.SUNKEN, highlightthickness=0)
        self.display_canvas.pack(fill=tk.X, pady=(0, 10))
        
        # --- Status Label ---
        self.status_label = ttk.Label(main_frame, text="Initializing...", font=("Helvetica", 10),
                                       background=BG_GREEN, anchor=tk.CENTER)
        self.status_label.pack(fill=tk.X, pady=(0, 10))

        # --- Button Frame ---
        button_frame = tk.Frame(main_frame, bg=BG_GREEN)
        button_frame.pack(fill=tk.X)

        # Configure columns in button_frame to space out buttons evenly
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        # --- Context Button ---
        self.context_button = tk.Button(button_frame, image=self.context_img_off, # Start OFF
                                        command=self.toggle_context, borderwidth=0,
                                        relief=tk.FLAT, bg=BG_GREEN, activebackground=BG_GREEN)
        self.context_button.grid(row=0, column=0, sticky=tk.W, padx=5) # Align left

        # --- Download Button ---
        self.download_button = tk.Button(button_frame, image=self.download_icon_disabled, # Start disabled
                                         command=self.download_audio, state=tk.DISABLED,
                                         borderwidth=0, relief=tk.FLAT, bg=BG_GREEN, activebackground=BG_GREEN)
        self.download_button.grid(row=0, column=1) # Centered

        # --- Record Button ---
        self.record_button = tk.Button(button_frame, image=self.record_icon_disabled, # Start disabled
                                       command=self.handle_read_context_request, state=tk.DISABLED,
                                       borderwidth=0, relief=tk.FLAT, bg=BG_GREEN, activebackground=BG_GREEN)
        self.record_button.grid(row=0, column=2, sticky=tk.E, padx=5) # Align right


        # ... (Rest of __init__, including DnD setup, backend init thread) ...
        self.is_animating_waveform = False
        self.after(100, self.animate_waveform) # Start animation loop

    def load_images(self):
        """Load all required images using Pillow."""
        try:
            # Context Button Images (Create these PNGs, ~40x25 pixels?)
            img_off_pil = Image.open("assets/context_off.png") # Put images in 'assets' folder
            self.context_img_off = ImageTk.PhotoImage(img_off_pil)
            img_on_pil = Image.open("assets/context_on.png")
            self.context_img_on = ImageTk.PhotoImage(img_on_pil)

            # Download Button Images (Create these PNGs, ~30x30 pixels?)
            img_dl_pil = Image.open("assets/download_icon.png")
            self.download_icon = ImageTk.PhotoImage(img_dl_pil)
            img_dl_disabled_pil = Image.open("assets/download_icon_disabled.png") # Greyscale version
            self.download_icon_disabled = ImageTk.PhotoImage(img_dl_disabled_pil)

            # Record Button Images (Create these PNGs, ~30x30 pixels?)
            img_rec_pil = Image.open("assets/record_icon.png") # Red circle
            self.record_icon = ImageTk.PhotoImage(img_rec_pil)
            img_rec_disabled_pil = Image.open("assets/record_icon_disabled.png") # Grey circle
            self.record_icon_disabled = ImageTk.PhotoImage(img_rec_disabled_pil)

            # Add active/pressed states if desired
            # img_rec_active_pil = Image.open("assets/record_icon_active.png")
            # self.record_icon_active = ImageTk.PhotoImage(img_rec_active_pil)
            # self.record_button.config(activeimage=self.record_icon_active) # Requires tk.Button

        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Failed to load image asset: {e}. Please ensure 'assets' folder exists.")
            # Handle error gracefully - maybe use text buttons as fallback?
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred loading images: {e}")
            self.destroy()

    # --- Update button states using images ---
    def set_processing_state(self, processing: bool):
        self.is_processing = processing

        def _update():
            record_img = self.record_icon_disabled
            record_state = tk.DISABLED
            context_state = tk.DISABLED if processing else tk.NORMAL # Context always toggleable unless busy?
            download_img = self.download_icon_disabled
            download_state = tk.DISABLED

            if not processing:
                if self.is_context_loaded:
                    record_img = self.record_icon # Use normal red icon
                    record_state = tk.NORMAL
                if self.last_tts_filepath:
                    download_img = self.download_icon # Use normal DL icon
                    download_state = tk.NORMAL

            self.record_button.config(image=record_img, state=record_state)
            self.context_button.config(state=context_state) # Image changes in toggle_context
            self.download_button.config(image=download_img, state=download_state)

        self.after(0, _update)


    def toggle_context(self):
         # ... (logic for clearing/loading context state) ...
         if self.is_context_loaded: # After loading via drop
             new_img = self.context_img_on
             # Enable record button (set_processing_state handles image/state)
         else: # When toggling OFF
             new_img = self.context_img_off
             # Disable record button (set_processing_state handles image/state)

         self.context_button.config(image=new_img)
         # Let set_processing_state handle record/download button states based on is_context_loaded
         self.set_processing_state(self.is_processing) # Re-evaluate button states


    def _load_context_thread(self, filepath):
        # ... load content ...
        try:
            # ... success ...
            self.is_context_loaded = True
            self.context_content = "..." # Loaded content
            self.after(0, lambda: self.context_button.config(image=self.context_img_on))
            # No need to explicitly enable record button here, set_processing_state will do it
        except:
            # ... error ...
            self.is_context_loaded = False
            self.after(0, lambda: self.context_button.config(image=self.context_img_off))
        finally:
            # Update button states based on new context status
             self.set_processing_state(False)


    # ... other methods (_initialize_backend, handle_speech_input etc.)

```

**Steps You Need to Take:**

1.  **Install Pillow:** `pip install Pillow`
2.  **Create Image Assets:** Design the following images (PNG format recommended, use transparency where needed) and save them in an `assets` subfolder relative to your script:
    *   `context_off.png`: Grey/black rectangle with border, maybe a grey inner bar.
    *   `context_on.png`: Green rectangle with border, green inner bar.
    *   `download_icon.png`: The download arrow icon.
    *   `download_icon_disabled.png`: A greyscale or faded version of the download icon.
    *   `record_icon.png`: The solid red circle.
    *   `record_icon_disabled.png`: A grey or faded version of the circle.
    *   *(Optional)* Active/pressed state images if desired.
3.  **Adjust Sizes:** Modify the `.resize((width, height))` values in `load_images` to match the actual dimensions of your created images.
4.  **Integrate:** Replace the conceptual code snippets into your main application class structure. Ensure image loading happens correctly in `load_images`. Update the `set_processing_state`, `toggle_context`, and `_load_context_thread` methods to configure the button `image` and `state` as shown.
5.  **Layout Tweaks:** Adjust `padx`, `pady`, widget sizes, and `sticky` options in `pack` or `grid` calls to fine-tune the positioning and spacing. Using `grid` within the `button_frame` provides more control over horizontal alignment.