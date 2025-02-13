import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def generate_checkerboard(size, color1, color2):

    rows, cols = 300, 500
    img = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if (i // (size // 8) + j // (size // 8)) % 2 == 0:
                img[i, j] = color1
            else:
                img[i, j] = color2
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

def plot_histogram(image):

    colors = ('b', 'g', 'r')  
    fig, ax = plt.subplots(figsize=(6, 3.24)) 

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.fill_between(range(256), hist.flatten(), color=color, alpha=0.5, label=f'{color.upper()} channel')

    ax.set_title("Color Histogram", fontsize=14, fontweight='bold')
    ax.set_xlabel("Pixel Intensity", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    return fig

def main():
    st.title("Warped Checkerboard with Enhanced Histogram")

    checkerboard_options = {
        "Classic (Black & White)": (400, (0, 0, 0), (255, 255, 255)),
        "Green & Purple": (400, (0, 255, 0), (128, 0, 128)),
        "Yellow & Cyan": (400, (255, 255, 0), (0, 255, 255)),
        "Blue & Red": (400, (0, 0, 255), (255, 255, 0)),
        "Custom": None  
    }

    selected_option = st.sidebar.selectbox("Choose a checkerboard style", list(checkerboard_options.keys()))

    if selected_option == "Custom":
        size = 400
        color1 = st.sidebar.color_picker("Select First Color", "#000000")
        color2 = st.sidebar.color_picker("Select Second Color", "#FFFFFF")
        color1 = tuple(int(color1.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        color2 = tuple(int(color2.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    else:
        size, color1, color2 = checkerboard_options[selected_option]

    warp_amplitude = st.sidebar.slider("Warping Intensity", min_value=0, max_value=30, value=0, step=1)

    checkerboard = generate_checkerboard(size, color1, color2)

    if warp_amplitude > 0:
        warped_checkerboard = warp_checkerboard(checkerboard, amplitude=warp_amplitude)
    else:
        warped_checkerboard = checkerboard

    mean, std_dev = calculate_mean_std(warped_checkerboard)

    formatted_mean = [f"{val:.2f}" for val in mean]
    formatted_std = [f"{val:.2f}" for val in std_dev]

    col1, col2 = st.columns([1, 1])

    with col1:
        caption = f"{selected_option} Checkerboard"
        caption += " (Warped)" if warp_amplitude > 0 else ""
        st.image(warped_checkerboard, caption=caption, channels="BGR", use_container_width=True)

    with col2:
        st.pyplot(plot_histogram(warped_checkerboard))

    st.sidebar.markdown("### Image Statistics:")
    st.sidebar.write(f"**Mean (B, G, R):**")
    st.sidebar.write(f"**{formatted_mean}**")
    st.sidebar.write(f"**Standard Deviation (B, G, R):**")
    st.sidebar.write(f"**{formatted_std}**")

if __name__ == "__main__":
    main()
