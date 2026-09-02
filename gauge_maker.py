import matplotlib.pyplot as plt
import numpy as np

def generate_risk_gauge(score, filename="risk_gauge.png"):
    # Setup the canvas
    fig, ax = plt.subplots(figsize=(6, 3))

    # Create the half-donut (Green, Yellow, Red, and a hidden bottom half)
    colors = ['#00B050', '#FFC000', '#FF0000', 'white']
    sizes = [33.3, 33.3, 33.4, 100] # Top half divided by 3, bottom half is 100 (hidden)

    # Draw the donut ring
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=180, # Start drawing from the left
        counterclock=False,
        wedgeprops={'width': 0.4, 'edgecolor': 'white'}
    )

    # Calculate the needle angle (Score 0 is left, 100 is right)
    angle = 180 - (score / 100) * 180
    theta = np.radians(angle)

    # Calculate the needle tip coordinates
    r = 0.6 # Length of the needle
    x_tip = r * np.cos(theta)
    y_tip = r * np.sin(theta)

    # Draw the needle pointing to the score
    ax.annotate(
        '',
        xy=(x_tip, y_tip), xycoords='data',
        xytext=(0, 0), textcoords='data',
        arrowprops=dict(arrowstyle="wedge,tail_width=0.6", color="#2C3E50", shrinkA=0, shrinkB=0)
    )

    # Print the large score text in the center
    ax.text(0, -0.15, f"{score}/100", horizontalalignment='center', verticalalignment='center', fontsize=24, fontweight='bold', color="#1A252F")
    ax.text(0, -0.35, "COMPLIANCE RISK", horizontalalignment='center', verticalalignment='center', fontsize=10, color='gray', fontweight='bold')

    # Clean up the visual and save it
    ax.axis('equal') 
    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Success! Risk Gauge saved as {filename}")

# ==========================================
# TEST THE CHART ENGINE
# ==========================================
if __name__ == "__main__":
    # Test it with a high-risk score
    generate_risk_gauge(85)