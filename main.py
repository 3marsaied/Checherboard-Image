import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(layout="wide")

def lerp(color_a, color_b, t):
    """Linearly interpolate between two colors."""
    return tuple(int((1 - t) * a + t * b) for a, b in zip(color_a, color_b))

def generate_checkerboard(size, color1, color2):
    """
    Generate a checkerboard that transitions from color1 to color2
    from left to right. Black squares remain black, while colored squares
    smoothly interpolate.
    """
    rows, cols = 200, 400
    img = np.zeros((rows, cols, 3), dtype=np.uint8)
    num_tiles = 8
    tile_size = size // num_tiles

    for i in range(rows):
        for j in range(cols):
            tile_row = i // tile_size
            tile_col = j // tile_size
            t = j / (cols - 1) if cols > 1 else 0  # global interpolation factor
            if (tile_row + tile_col) % 2 == 0:
                pixel_color = (0, 0, 0)  # black
            else:
                pixel_color = lerp(color1, color2, t)
            img[i, j] = pixel_color
    return img

def warp_checkerboard(image, amplitude=10):
    rows, cols, _ = image.shape
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            map_x[i, j] = j + amplitude * np.sin(2 * np.pi * i / 60)
            map_y[i, j] = i + amplitude * np.sin(2 * np.pi * j / 60)
    warped_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped_img

def calculate_mean_std(image):
    mean, std_dev = cv2.meanStdDev(image)
    return mean.flatten(), std_dev.flatten()

def plot_histogram_for_channel(image, channel_index, channel_name):
    """
    Plot histogram for a single channel (Blue, Green, or Red) using Plotly.
    """
    hist = cv2.calcHist([image], [channel_index], None, [256], [0, 256]).flatten()
    fig = go.Figure(
        data=[go.Bar(
            x=list(range(256)),
            y=hist,
            marker_color=channel_name.lower(),
            opacity=0.6
        )]
    )
    fig.update_layout(
        title=f"{channel_name} Channel Histogram",
        xaxis_title="Pixel Intensity",
        yaxis_title="Frequency",
        xaxis=dict(range=[0, 256]),
        yaxis=dict(range=[0, 2500]),
        template="plotly_white",
        height=400  # Ensuring consistent height with the image
    )
    return fig

def main():
    st.title("Warped Gradient Checkerboard with Plotly Histograms")

    # Default color pairs (BGR)
    checkerboard_options = {
        "Pink -> Yellow": (400, (203, 192, 255), (0, 255, 255)),  # BGR for pink -> yellow
        "Blue -> Red": (400, (255, 0, 0), (0, 0, 255)),
        "Green -> Purple": (400, (0, 255, 0), (128, 0, 128)),
    }

    # Select one of the predefined options
    selected_option = st.sidebar.selectbox("Choose a checkerboard style", list(checkerboard_options.keys()))
    size, color1, color2 = checkerboard_options[selected_option]

    # Warp intensity slider
    warp_amplitude = st.sidebar.slider("Warping Intensity", min_value=0, max_value=30, value=0, step=1)

    # Generate and optionally warp the checkerboard
    checkerboard = generate_checkerboard(size, color1, color2)
    if warp_amplitude > 0:
        warped_checkerboard = warp_checkerboard(checkerboard, amplitude=warp_amplitude)
    else:
        warped_checkerboard = checkerboard

    # Compute statistics
    mean, std_dev = calculate_mean_std(warped_checkerboard)
    formatted_mean = [f"{val:.2f}" for val in mean]
    formatted_std = [f"{val:.2f}" for val in std_dev]

    # Convert checkerboard to PIL Image for better scaling
    checkerboard_pil = Image.fromarray(cv2.cvtColor(warped_checkerboard, cv2.COLOR_BGR2RGB))

    # Display the checkerboard and histograms in equal height
    col1, col2 = st.columns([1, 1])
    with col1:
        caption = f"{selected_option} Checkerboard" + (" (Warped)" if warp_amplitude > 0 else "")
        st.image(checkerboard_pil, caption=caption, use_column_width=True)  # Make it as wide as histograms
        st.plotly_chart(plot_histogram_for_channel(warped_checkerboard, 1, "Green"), use_container_width=True)

    with col2:
        st.plotly_chart(plot_histogram_for_channel(warped_checkerboard, 2, "Red"), use_container_width=True)
        st.plotly_chart(plot_histogram_for_channel(warped_checkerboard, 0, "Blue"), use_container_width=True)

    # Display statistics in sidebar
    st.sidebar.markdown("### Image Statistics:")
    st.sidebar.write(f"**Mean (B, G, R):** {formatted_mean}")
    st.sidebar.write(f"**Standard Deviation (B, G, R):** {formatted_std}")

if __name__ == "__main__":
    main()
