# ui/review_window.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Explicitly use Tkinter backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp # For connections
import traceback

# Import constants directly
import config

class SampleReviewWindow(tk.Toplevel):
    def __init__(self, parent_app, gesture_name, samples_copy):
        """
        Initializes the Sample Review window.
        Args:
            parent_app: The main HandGestureRecognizer instance.
            gesture_name: The name of the gesture being reviewed.
            samples_copy: A COPY of the list of samples for this gesture.
        """
        super().__init__(parent_app.root) # Parent is the main root window
        self.parent_app = parent_app
        self.gesture_name = gesture_name
        self.samples = samples_copy # Work on the copy
        self.current_index = 0 if self.samples else -1
        self.initial_sample_count = len(self.samples)

        self.title(f"Review Samples: {self.gesture_name}")
        self.geometry("500x680"); self.minsize(450, 580); self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close_prompt)

        # --- Main Frame ---
        main_frame = ttk.Frame(self, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(0, weight=1); main_frame.grid_rowconfigure(1, weight=1) # Canvas expands

        # --- Info Label ---
        self.info_label = ttk.Label(main_frame, text="", anchor=tk.CENTER, font=('TkDefaultFont', 11, 'bold'))
        self.info_label.grid(row=0, column=0, pady=(0, 5), sticky="ew")

        # --- Matplotlib Setup ---
        plot_bg_color = self._get_safe_matplotlib_bg_color()
        self.fig = plt.figure(figsize=(5, 4.5), dpi=90, facecolor=plot_bg_color)
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor=plot_bg_color)
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98) # Minimize padding
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, pady=5, sticky="nsew")

        # --- Matplotlib Toolbar ---
        toolbar_frame = ttk.Frame(main_frame); toolbar_frame.grid(row=2, column=0, sticky="ew", pady=(0,5))
        try: self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame); self.toolbar.update()
        except Exception as e: print(f"Info: Matplotlib Toolbar disabled ({e}).")

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=3, column=0, pady=(5, 5), sticky="ew")

        # --- Navigation Controls ---
        nav_controls_frame = ttk.Frame(main_frame); nav_controls_frame.grid(row=4, column=0, pady=5, sticky="ew"); nav_controls_frame.grid_columnconfigure(1, weight=1)
        self.prev_button = ttk.Button(nav_controls_frame, text="< Prev", command=self._show_previous, width=10); self.prev_button.grid(row=0, column=0, padx=(0, 5))
        self.goto_var = tk.StringVar(); self.goto_entry = ttk.Entry(nav_controls_frame, textvariable=self.goto_var, width=6, justify='center'); self.goto_entry.grid(row=0, column=1, padx=5, sticky="ew"); self.goto_entry.bind("<Return>", lambda e: self._go_to_sample())
        self.goto_button = ttk.Button(nav_controls_frame, text="Go", command=self._go_to_sample, width=4); self.goto_button.grid(row=0, column=2, padx=(0, 5))
        self.next_button = ttk.Button(nav_controls_frame, text="Next >", command=self._show_next, width=10); self.next_button.grid(row=0, column=3, padx=(5, 0))

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=5, column=0, pady=(5, 5), sticky="ew")

        # --- Action Frame ---
        action_frame = ttk.Frame(main_frame); action_frame.grid(row=6, column=0, pady=(10, 0), sticky="ew"); action_frame.grid_columnconfigure(0, weight=1); action_frame.grid_columnconfigure(1, weight=1)
        style = ttk.Style(self); style.configure("Delete.TButton", foreground="red", font=('TkDefaultFont', 9, 'bold'))
        self.delete_button = ttk.Button(action_frame, text="❌ Delete Sample", command=self._delete_current, style="Delete.TButton"); self.delete_button.grid(row=0, column=0, pady=5, padx=(0, 5), sticky="ew")
        self.save_button = ttk.Button(action_frame, text="💾 Save & Close", command=self._save_and_close); self.save_button.grid(row=0, column=1, pady=5, padx=(5, 0), sticky="ew")

        self._update_display() # Initial draw and button state update
        self.lift(); self.focus_force() # Bring to front and give focus

    def _get_safe_matplotlib_bg_color(self):
        """Safely gets the system background color for Matplotlib."""
        fallback_color = "#F0F0F0" # Standard Tkinter light gray
        try:
            # Ensure master (the root window passed during Toplevel init) exists
            if self.master and isinstance(self.master, (tk.Tk, tk.Toplevel)):
                style = ttk.Style(self.master)
                bg = style.lookup('TFrame', 'background')
                # Validate color before returning
                if matplotlib.colors.is_color_like(bg):
                    return bg
                else: print(f"Info: Sys BG '{bg}' not usable by Matplotlib.")
            else: print("Info: Invalid master for style lookup.")
        except tk.TclError: print("Warn: TclError getting BG color.")
        except Exception as e: print(f"Warn: Error getting BG color: {e}")
        return fallback_color

    def _draw_current_sample_3d(self):
        """Draws the current sample onto the 3D Axes."""
        try:
            self.ax.clear()
            if not self.samples or self.current_index < 0 or self.current_index >= len(self.samples):
                self.ax.text(0.5, 0.5, 0.5, "No Samples Remaining", ha='center', va='center', color='gray', transform=self.ax.transAxes); return

            sample_features = self.samples[self.current_index]
            # Expecting 69 features, need at least the first 60 for coords
            if not isinstance(sample_features, (list, np.ndarray)) or len(sample_features) < 60:
                self.ax.text(0.5, 0.5, 0.5, "Invalid Sample Format", ha='center', va='center', color='red', transform=self.ax.transAxes);
                print(f"Warn: Invalid sample format idx {self.current_index} for {self.gesture_name}"); return

            # --- Reconstruct Absolute Coordinates for Visualization ---
            # IMPORTANT: The *features* are relative and normalized. We need something
            # visually consistent. Since we don't store the original scale factor
            # or absolute wrist pos, we'll visualize the RELATIVE coordinates
            # directly. This means the scale won't match real world, but the shape will.
            coords_xyz_relative = [(0.0, 0.0, 0.0)] # Wrist at origin for visualization
            feature_idx = 0
            for _ in range(20): # Read the first 60 features (relative coords)
                if feature_idx + 2 >= 60: print(f"Error: Feature list < 60 for sample {self.current_index}."); self.ax.text(0.5,0.5,0.5,"Feature Len Error"); return
                # Use the stored (potentially normalized) relative coords for VISUALIZATION
                x = sample_features[feature_idx]
                y = sample_features[feature_idx + 1]
                z = sample_features[feature_idx + 2]
                coords_xyz_relative.append((x, y, z)); feature_idx += 3

            if len(coords_xyz_relative) != 21: self.ax.text(0.5,0.5,0.5,"Coord Count Error"); return

            x_vals, y_vals, z_vals = zip(*coords_xyz_relative)

            # Plot connections
            connections = mp.solutions.hands.HAND_CONNECTIONS
            if connections:
                for start_idx, end_idx in connections:
                    if 0 <= start_idx < 21 and 0 <= end_idx < 21:
                        self.ax.plot([x_vals[start_idx], x_vals[end_idx]], [y_vals[start_idx], y_vals[end_idx]], [z_vals[start_idx], z_vals[end_idx]], color=config.LINE_COLOR_3D, linewidth=2.5, alpha=0.8)

            # Plot points
            self.ax.scatter(x_vals[1:], y_vals[1:], z_vals[1:], c=config.POINT_COLOR_3D, s=config.POINT_SIZE, depthshade=True, alpha=0.9) # Other points blue
            self.ax.scatter(x_vals[0], y_vals[0], z_vals[0], c=config.WRIST_COLOR_3D, s=config.POINT_SIZE*1.5, depthshade=True, marker='s', alpha=1.0) # Wrist red square

            # Set plot limits and aspect ratio - make robust
            try:
                x_min, x_max = min(x_vals), max(x_vals); y_min, y_max = min(y_vals), max(y_vals); z_min, z_max = min(z_vals), max(z_vals)
                x_range = x_max-x_min; y_range = y_max-y_min; z_range = z_max-z_min
                # Handle cases where range is zero
                if x_range < 1e-6: x_range = 0.1; x_min -= 0.05; x_max += 0.05
                if y_range < 1e-6: y_range = 0.1; y_min -= 0.05; y_max += 0.05
                if z_range < 1e-6: z_range = 0.1; z_min -= 0.05; z_max += 0.05
                max_dim_range = max(x_range, y_range, z_range); buffer = max_dim_range * 0.1
                mid_x=(x_max+x_min)/2.0; mid_y=(y_max+y_min)/2.0; mid_z=(z_max+z_min)/2.0
                # Set equal aspect ratio centered around the middle point
                self.ax.set_xlim(mid_x - max_dim_range/2 - buffer, mid_x + max_dim_range/2 + buffer)
                self.ax.set_ylim(mid_y - max_dim_range/2 - buffer, mid_y + max_dim_range/2 + buffer)
                self.ax.set_zlim(mid_z - max_dim_range/2 - buffer, mid_z + max_dim_range/2 + buffer)
            except ValueError as ve: print(f"Error calculating limits: {ve}"); self.ax.auto_scale_xyz([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]) # Fallback

            # Clean up Axes
            self.ax.set_xlabel("X (Relative)"); self.ax.set_ylabel("Y (Relative)"); self.ax.set_zlabel("Z (Relative)")
            self.ax.grid(False); self.ax.xaxis.pane.fill=False; self.ax.yaxis.pane.fill=False; self.ax.zaxis.pane.fill=False
            self.ax.xaxis.pane.set_edgecolor('w'); self.ax.yaxis.pane.set_edgecolor('w'); self.ax.zaxis.pane.set_edgecolor('w')
            self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_zticks([])
            # Set orientation (adjust azim for better side view)
            self.ax.invert_yaxis(); self.ax.invert_zaxis() # Match MediaPipe convention
            self.ax.view_init(elev=20., azim=-75) # Adjusted azimuth angle
            try: self.fig.tight_layout(pad=0.1) # Adjust padding if needed
            except Exception as e: print(f"Warn: tight_layout failed: {e}")
        except Exception as e:
            print(f"--- Error in _draw_current_sample_3d ---"); traceback.print_exc()
            self.ax.clear(); self.ax.text(0.5,0.5,0.5,"Error Drawing Sample", ha='center', va='center', color='red', transform=self.ax.transAxes)
        finally:
            try: self.canvas.draw_idle() # Ensure canvas update
            except Exception as e: print(f"Error drawing canvas: {e}")

    def _update_display(self):
        """Updates labels, button states, and redraws the 3D plot."""
        num_samples = len(self.samples); current_display_num = self.current_index + 1 if num_samples > 0 else 0
        if num_samples == 0: self.info_label.config(text=f"{self.gesture_name} - No Samples"); self.current_index = -1
        else: self.info_label.config(text=f"{self.gesture_name} - Sample {current_display_num} / {num_samples}")
        self._draw_current_sample_3d()
        is_valid_index = 0 <= self.current_index < num_samples; changes_made = len(self.samples) != self.initial_sample_count
        self.prev_button.config(state=tk.NORMAL if is_valid_index and self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if is_valid_index and self.current_index < num_samples - 1 else tk.DISABLED)
        self.delete_button.config(state=tk.NORMAL if is_valid_index else tk.DISABLED)
        self.goto_button.config(state=tk.NORMAL if num_samples > 0 else tk.DISABLED); self.goto_entry.config(state=tk.NORMAL if num_samples > 0 else tk.DISABLED)
        self.save_button.config(state=tk.NORMAL if changes_made else tk.DISABLED) # Enable save only if changes were made

    def _show_next(self):
        num_samples = len(self.samples);
        if num_samples > 0 and self.current_index < num_samples - 1: self.current_index += 1; self._update_display()

    def _show_previous(self):
        if self.current_index > 0: self.current_index -= 1; self._update_display()

    def _go_to_sample(self):
        num_samples = len(self.samples);
        if num_samples <= 0: return
        try:
            target_sample_num = int(self.goto_var.get())
            if 1 <= target_sample_num <= num_samples: self.current_index = target_sample_num - 1; self._update_display(); self.goto_var.set("")
            else: messagebox.showerror("Invalid Number", f"Enter num 1-{num_samples}.", parent=self); self.goto_var.set("")
        except ValueError: messagebox.showerror("Invalid Input", "Enter number.", parent=self); self.goto_var.set("")
        except Exception as e: messagebox.showerror("Error", f"Err: {e}", parent=self); print(f"GoTo Err: {e}"); traceback.print_exc(); self.goto_var.set("")

    def _delete_current(self):
        num_samples = len(self.samples);
        if num_samples <= 0 or self.current_index < 0: return
        # Confirmation dialog
        confirm = messagebox.askyesno("Confirm Delete", f"Delete sample {self.current_index + 1} for '{self.gesture_name}'?", parent=self)
        if confirm:
            try:
                del self.samples[self.current_index]
                # Adjust index AFTER deletion
                if self.current_index >= len(self.samples): # If last item was deleted
                    self.current_index = max(-1, len(self.samples) - 1) # Go to new last item or -1 if empty
                # No need to change index if deleting from middle or beginning

                if len(self.samples) == 0: self.current_index = -1 # Explicitly handle empty list
                self._update_display(); print(f"Deleted sample for {self.gesture_name}. Rem: {len(self.samples)}")
            except IndexError:
                 print("Error: Index out of bounds during deletion (should not happen).")
                 self._update_display() # Refresh display even on error

    def _save_and_close(self):
        """Applies changes to the parent app's data and closes."""
        if self.parent_app:
            num_deleted = self.initial_sample_count - len(self.samples); print(f"Saving review for {self.gesture_name}. Del {num_deleted}.")
            # Update the actual data in the parent app
            self.parent_app.data[self.gesture_name] = self.samples
            # Use the parent's save method
            success, _, status_msg = self.parent_app.save_data_external() # Use external save helper
            if success:
                self.parent_app.update_data_list(); final_status = f"Reviewed '{self.gesture_name}', {len(self.samples)} saved."
                if num_deleted > 0: self.parent_app.invalidate_model(); final_status += " Retrain needed."
                self.parent_app.set_status(final_status)
            else: messagebox.showerror("Save Error", f"Failed save:\n{status_msg}", parent=self); return # Show error and stay
        self.destroy() # Close the window

    def _on_close_prompt(self):
         """Handles the close (X) button click."""
         num_deleted = self.initial_sample_count - len(self.samples)
         if num_deleted > 0: # Only prompt if changes were made (samples deleted)
             confirm = messagebox.askyesnocancel("Unsaved Changes", f"Deleted {num_deleted} sample(s).\nSave changes before closing?", parent=self)
             if confirm is True: self._save_and_close() # Save and then close
             elif confirm is False: self.destroy() # Discard changes and close
             # else: Do nothing if Cancelled, window stays open
         else:
             self.destroy() # No changes, just close