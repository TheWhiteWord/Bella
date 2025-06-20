"""

**1. Making the Waves More Prominent**

To make the waves more visually striking, you can adjust a few parameters in the script:

*   **`ripple_amplitude_max_demo`**: This controls the maximum displacement of the pixels in the ripple. Increasing it will make the wave distortion more significant.
    *   *Suggestion*: Change from `20.0` to `35.0` (or even higher if you want a very strong effect).
*   **`wave_layer_opacity_factor`**: This controls the opacity of the green wave layer itself before it's blended. Increasing it (making it closer to 1.0) will make the green color of the wave more solid and visible.
    *   *Suggestion*: Change from `0.39` (39%) to `0.55` (55%).
*   **`target_lightness_percent`** (for the green wave color): This is part of the HSL color transformation. Making the green wave inherently brighter will make it stand out more, especially with the "Screen" blend mode.
    *   *Suggestion*: Change from `14.20` to `20.0` or `25.0`.

**2. Adding `input_file_3.png` as a Screen Overlay**

You want to add `input_file_3.png` (which has shading and light effects) on top, making the wave animation appear *underneath* this screen. This is a great idea to add depth!

*   We'll load `input_file_3.png`.
*   After the sphere and the animated wave are combined, we will composite this new "screen overlay" image on top.
*   You mentioned "opacity and layers type". For "layers type", a common choice for such screen effects (lights/shadows) is the "Overlay" blend mode. "Normal" blend mode with adjusted opacity is also an option if `input_file_3.png` is designed with its own transparency for this. I'll implement it using the "Overlay" blend mode, which should interact nicely with the lights and shadows in your overlay image.
*   We'll add a parameter for the opacity of this screen overlay layer, e.g., `screen_overlay_opacity_factor`.

"""
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageChops
# colorsys is not strictly needed if we skip HSL transform, but keep for function structure
import colorsys 
import os

# --- Configuration based on user input ---
sphere_frame_img_path = "frame.png"
# This is now the "monster mouth" image, used more directly
wave_source_material_path = "wave.png" 
screen_overlay_img_path = "screen.png"

# These HSL/Brightness parameters will be bypassed if using wave_source_material directly
target_hue_degrees = 130.33
target_chroma_percent = 59.98
target_lightness_percent = 22.0
brightness_factor_from_user = -83.54
actual_brightness_factor = (100.0 + brightness_factor_from_user) / 100.0

# Ripple parameters
ripple_amplitude_max_demo = 60.0 
ripple_period_user = 41.4
ripple_initial_phase_user = 0.593

# Wave layer opacity (applied to the input_file_0.png directly)
wave_layer_opacity_percent = 55.0 
wave_layer_opacity_factor = wave_layer_opacity_percent / 100.0

# Screen Overlay Layer parameters
screen_overlay_opacity_percent = 75.0 
screen_overlay_opacity_factor = screen_overlay_opacity_percent / 100.0
screen_overlay_blend_mode = "overlay" 

# Animation parameters
num_frames = 30
duration_ms_per_frame = 100
output_gif_path = "animated_voice_wave_final.gif" # New output name

# --- Helper Functions (transform_color_hsl_optimized and apply_brightness might not be used) ---
def apply_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def transform_color_hsl_optimized(image, target_h_deg, target_s_pct, target_l_pct):
    target_h = target_h_deg / 360.0
    target_s = target_s_pct / 100.0
    target_l = target_l_pct / 100.0
    img_rgba = image.convert("RGBA")
    img_arr = np.array(img_rgba, dtype=float) / 255.0
    r, g, b, a = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2], img_arr[:,:,3]
    hls_array = np.empty_like(img_arr[:,:,:3])
    for i_row in range(r.shape[0]):
        for j_col in range(r.shape[1]):
            if a[i_row,j_col] < 1e-6 :
                 hls_array[i_row,j_col,:] = colorsys.rgb_to_hls(r[i_row,j_col], g[i_row,j_col], b[i_row,j_col])
                 continue
            hls_array[i_row,j_col,:] = colorsys.rgb_to_hls(r[i_row,j_col], g[i_row,j_col], b[i_row,j_col])
    hls_array[:,:,0] = target_h
    hls_array[:,:,1] = target_l
    hls_array[:,:,2] = target_s
    rgb_array_transformed = np.empty_like(img_arr[:,:,:3])
    for i_row in range(r.shape[0]):
        for j_col in range(r.shape[1]):
            if a[i_row,j_col] < 1e-6: 
                rgb_array_transformed[i_row,j_col,:] = [r[i_row,j_col],g[i_row,j_col],b[i_row,j_col]]
                continue
            rgb_array_transformed[i_row,j_col,:] = colorsys.hls_to_rgb(hls_array[i_row,j_col,0], hls_array[i_row,j_col,1], hls_array[i_row,j_col,2])
    final_rgb_arr = (np.clip(rgb_array_transformed, 0, 1) * 255.0).astype(np.uint8)
    alpha_arr = (a * 255.0).astype(np.uint8)
    transformed_arr = np.dstack((final_rgb_arr, alpha_arr))
    return Image.fromarray(transformed_arr, 'RGBA')

def apply_ripple_effect_optimized(image, amplitude, period, phase_shift):
    img_rgba = image.convert("RGBA") # Ensure RGBA for safety
    img_arr = np.array(img_rgba)
    height, width = img_arr.shape[:2]
    y_coords, x_coords = np.ogrid[:height, :width]
    y_displacement = amplitude * np.sin(2 * np.pi * x_coords / period + phase_shift)
    y_source_float = y_coords - y_displacement
    y_source = np.clip(y_source_float, 0, height - 1).astype(int)
    x_source = x_coords
    output_arr = img_arr[y_source, x_source]
    return Image.fromarray(output_arr, 'RGBA')

def screen_blend_effect(base_img_rgba, top_img_rgba_effect):
    base_rgb = base_img_rgba.convert("RGB")
    top_rgb = top_img_rgba_effect.convert("RGB")
    blended_rgb = ImageChops.screen(base_rgb, top_rgb)
    screened_effect_layer_rgba = blended_rgb.convert("RGBA")
    screened_effect_layer_rgba.putalpha(top_img_rgba_effect.split()[-1])
    return screened_effect_layer_rgba

def set_opacity(image, opacity_factor):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    alpha = image.split()[-1]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity_factor)
    image_copy = image.copy()
    image_copy.putalpha(alpha)
    return image_copy

def apply_custom_blend_mode(base_rgba, top_rgba, mode, opacity=1.0):
    base = base_rgba.convert("RGBA")
    top = top_rgba.convert("RGBA") # Ensure top is RGBA

    if opacity < 1.0:
        top = set_opacity(top, opacity) # Use set_opacity for clarity

    base_rgb = base.convert("RGB")
    top_rgb = top.convert("RGB")

    blended_rgb = None
    if mode.lower() == 'multiply':
        blended_rgb = ImageChops.multiply(base_rgb, top_rgb)
    elif mode.lower() == 'screen':
        blended_rgb = ImageChops.screen(base_rgb, top_rgb)
    elif mode.lower() == 'overlay':
        base_arr = np.array(base_rgb, dtype=float) / 255.0
        top_arr = np.array(top_rgb, dtype=float) / 255.0
        res_arr = np.empty_like(base_arr)
        for i in range(3):
            b_ch, t_ch = base_arr[:,:,i], top_arr[:,:,i]
            res_arr[:,:,i] = np.where(b_ch <= 0.5, 2 * b_ch * t_ch, 1 - 2 * (1 - b_ch) * (1 - t_ch))
        blended_rgb = Image.fromarray((np.clip(res_arr, 0, 1) * 255.0).astype(np.uint8), "RGB")
    elif mode.lower() == 'normal':
        return Image.alpha_composite(base, top)
    else:
        raise ValueError(f"Unsupported blend mode: {mode}")

    blended_effect_rgba = blended_rgb.convert("RGBA")
    blended_effect_rgba.putalpha(top.split()[3])
    
    return Image.alpha_composite(base, blended_effect_rgba)

# --- Main Animation Logic ---
try:
    print("Loading base images...")
    sphere_frame = Image.open(sphere_frame_img_path).convert("RGBA")
    # wave_source_material is the "monster mouth" image
    wave_source_material = Image.open(wave_source_material_path).convert("RGBA") 
    screen_overlay_img = Image.open(screen_overlay_img_path).convert("RGBA")

    target_size = sphere_frame.size
    if wave_source_material.size != target_size:
        print(f"Resizing wave source material from {wave_source_material.size} to {target_size}")
        wave_source_material = wave_source_material.resize(target_size, Image.Resampling.LANCZOS)
    if screen_overlay_img.size != target_size:
        print(f"Resizing screen overlay image from {screen_overlay_img.size} to {target_size}")
        screen_overlay_img = screen_overlay_img.resize(target_size, Image.Resampling.LANCZOS)

    # --- Using wave_source_material (monster mouth) directly as the base for the wave ---
    # The previous steps of apply_brightness and transform_color_hsl are skipped.
    # We assume wave_source_material is already the desired color and has appropriate alpha.
    base_wave_image = wave_source_material 
    print("Using input_file_0.png directly as wave base.")

    frames = []
    print(f"Generating {num_frames} frames for animation...")
    for i in range(num_frames):
        progress = i / float(num_frames)
        current_amplitude = ripple_amplitude_max_demo * math.sin(math.pi * progress)
        current_phase_shift = ripple_initial_phase_user + (progress * 2 * math.pi * 2)

        # 1. Create animated rippled layer from base_wave_image
        rippled_wave_layer = apply_ripple_effect_optimized(base_wave_image, 
                                                 current_amplitude, 
                                                 ripple_period_user, 
                                                 current_phase_shift)
        # Set overall opacity of the rippled layer
        rippled_wave_layer_with_opacity = set_opacity(rippled_wave_layer, wave_layer_opacity_factor)

        # 2. Blend this wave effect onto the sphere_frame
        wave_effect_to_composite = screen_blend_effect(sphere_frame, rippled_wave_layer_with_opacity)
        sphere_with_wave = Image.alpha_composite(sphere_frame.copy(), wave_effect_to_composite)

        # 3. Apply the screen_overlay_img on top of the sphere_with_wave
        final_frame = apply_custom_blend_mode(sphere_with_wave, 
                                              screen_overlay_img, 
                                              screen_overlay_blend_mode, 
                                              screen_overlay_opacity_factor)
        
        # --- CRITICAL CHANGE FOR TRANSPARENCY ---
        # Append the RGBA frame directly, do not convert to RGB
        frames.append(final_frame) 
        # --- End of critical change ---

        if (i+1)%5 == 0 or i == num_frames -1 :
             print(f"Frame {i+1}/{num_frames} generated.")
             
    print("Saving animated GIF...")
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms_per_frame,
        loop=0,
        optimize=True,
        transparency=0 if frames[0].mode == 'P' else None, # Attempt to handle transparency better
        disposal=2 # Dispose of previous frame to allow transparency
    )
    print(f"Animated GIF saved as {output_gif_path}")

except FileNotFoundError as e:
    print(f"Error: File not found. Please ensure these files are in the directory.")
    print(f"Details: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()