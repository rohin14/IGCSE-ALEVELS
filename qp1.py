import streamlit as st
import pandas as pd
import json
import os
import time
import random
import re
from datetime import datetime
import tempfile
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import matplotlib.pyplot as plt
import numpy as np
import base64
from PIL import Image as PILImage, ImageDraw, ImageFont
import math
from dotenv import load_dotenv
import os

# Set up page configuration
st.set_page_config(
    page_title="ExamPrep AI - Past Paper Generator",
    page_icon="📝",
    layout="wide"
)


load_dotenv()  # Loads .env into environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 # Replace this with your actual Groq API key

# Initialize session state variables if they don't exist
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'selected_subject' not in st.session_state:
    st.session_state.selected_subject = None
if 'selected_topics' not in st.session_state:
    st.session_state.selected_topics = []

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F7FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 1rem;
        color: #333333;
    }
    .question-box {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
        color: #333333;
    }
    .mark-scheme {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #047857;
        margin-top: 0.5rem;
        color: #333333;
    }
    .diagram-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #666;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        color: #333333;
    }
    .difficulty-easy {
        color: #047857;
        font-weight: bold;
    }
    .difficulty-medium {
        color: #B45309;
        font-weight: bold;
    }
    .difficulty-hard {
        color: #B91C1C;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">ExamPrep AI: Past Paper Question Generator</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
Generate custom exam-style questions based on real IGCSE and A-Level past papers. Practice with 
AI-generated questions that match the format, difficulty, and style of actual exam questions.
</div>
""", unsafe_allow_html=True)

# Subject and curriculum data
SUBJECTS = {
    "IGCSE": [
        "Mathematics", "Physics", "Chemistry", "Biology", 
        "English Language", "English Literature", "Computer Science",
        "Business Studies", "Economics", "Geography", "History"
    ],
    "A-Level": [
        "Mathematics", "Further Mathematics", "Physics", "Chemistry", "Biology",
        "English Literature", "Computer Science", "Economics",
        "Business", "Psychology", "Sociology", "Geography", "History"
    ]
}

TOPICS = {
    "IGCSE Mathematics": [
        "Number", "Algebra", "Geometry", "Statistics and Probability", 
        "Functions", "Vectors and Transformations", "Calculus"
    ],
    "IGCSE Physics": [
        "Mechanics", "Thermal Physics", "Waves", "Electricity and Magnetism", 
        "Modern Physics", "Energy", "Radioactivity"
    ],
    "IGCSE Chemistry": [
        "Atomic Structure", "Bonding", "Periodic Table", "Chemical Reactions", 
        "Acids and Bases", "Organic Chemistry", "Quantitative Chemistry"
    ],
    "IGCSE Biology": [
        "Cell Biology", "Human Biology", "Plant Biology", "Ecology", 
        "Genetics", "Evolution", "Microbiology"
    ],
    "A-Level Mathematics": [
        "Pure Mathematics", "Calculus", "Mechanics", "Statistics", 
        "Probability", "Vectors", "Differential Equations"
    ],
    "A-Level Physics": [
        "Mechanics", "Materials", "Waves", "Electricity", "Magnetism", 
        "Nuclear Physics", "Particle Physics", "Quantum Physics", "Thermodynamics"
    ],
    "A-Level Chemistry": [
        "Physical Chemistry", "Inorganic Chemistry", "Organic Chemistry", 
        "Analytical Chemistry", "Thermodynamics", "Electrochemistry", "Kinetics"
    ],
    "A-Level Biology": [
        "Cell Biology", "Molecular Biology", "Genetics", "Ecology", 
        "Human Physiology", "Plant Biology", "Evolution", "Biochemistry"
    ]
}

# Available Groq models
GROQ_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it"
]

# Sample question formats
QUESTION_FORMATS = {
    "Multiple Choice": "Generate a multiple choice question with 4 options (A, B, C, D) and one correct answer.",
    "Short Answer": "Generate a question requiring a short answer (1-2 sentences).",
    "Calculation": "Generate a question requiring mathematical calculation and working.",
    "Extended Response": "Generate a question requiring an extended response (paragraph or essay).",
    "Practical": "Generate a question about experimental design or interpretation of results."
}


# Function to process diagram descriptions
def process_diagram_text(text):
    """
    Extract diagram descriptions from text and return both the clean text
    and a list of diagram descriptions.
    """
    # Pattern to identify diagram descriptions - adjust based on your model's output format
    diagram_pattern = r'\[DIAGRAM:([^\]]+)\]'
    
    # Find all diagram descriptions
    diagrams = re.findall(diagram_pattern, text)
    
    # Replace diagram placeholders with numbered references
    clean_text = re.sub(diagram_pattern, lambda m: f"[See Diagram {diagrams.index(m.group(1))+1}]", text)
    
    return clean_text, diagrams


# Function to generate diagrams based on description
def generate_diagram(description, index, width=600, height=400):
    """
    Generate a more visual diagram based on the description.
    Analyzes the text to determine what kind of diagram to create.
    """
    # Convert description to lowercase for easier matching
    desc_lower = description.lower()
    
    # Default is a simple text diagram
    if "graph" in desc_lower or "plot" in desc_lower or "curve" in desc_lower:
        return generate_graph_diagram(description, index, width, height)
    elif "circuit" in desc_lower:
        return generate_circuit_diagram(description, index, width, height)
    elif "triangle" in desc_lower or "square" in desc_lower or "circle" in desc_lower or "angle" in desc_lower:
        return generate_geometric_diagram(description, index, width, height)
    elif "cell" in desc_lower or "organ" in desc_lower or "plant" in desc_lower or "animal" in desc_lower:
        return generate_biology_diagram(description, index, width, height)
    elif "molecule" in desc_lower or "atom" in desc_lower or "compound" in desc_lower or "reaction" in desc_lower:
        return generate_chemistry_diagram(description, index, width, height)
    else:
        return generate_text_diagram(description, index, width, height)


def generate_text_diagram(description, index, width=600, height=400):
    """Create a basic text diagram"""
    # Create a blank image with a white background
    img = PILImage.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw a border
    draw.rectangle([(10, 10), (width-10, height-10)], outline='black', width=2)
    
    # Add a title
    draw.text((width//2-80, 20), f"Diagram {index}", fill='black', font=title_font)
    
    # Wrap and draw the description text
    lines = []
    words = description.split()
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        est_width = len(test_line) * 10  # Rough estimate of text width
        
        if est_width < width - 60:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw each line of text
    y_position = 80
    for line in lines:
        draw.text((30, y_position), line, fill='black', font=font)
        y_position += 25
    
    # Save to BytesIO
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def generate_graph_diagram(description, index, width=600, height=400):
    """Create a graph or plot based on the description"""
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    # Determine the type of graph from the description
    desc_lower = description.lower()
    
    if "bar" in desc_lower or "histogram" in desc_lower:
        # Generate a bar chart
        categories = ['A', 'B', 'C', 'D', 'E']
        values = np.random.randint(10, 100, size=5)
        ax.bar(categories, values)
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        ax.set_title(f'Diagram {index}: Bar Chart')
        
    elif "pie" in desc_lower:
        # Generate a pie chart
        labels = ['Category A', 'Category B', 'Category C', 'Category D']
        sizes = np.random.rand(4)
        sizes = sizes / sizes.sum()  # Normalize to sum to 1
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title(f'Diagram {index}: Pie Chart')
        
    elif "scatter" in desc_lower:
        # Generate a scatter plot
        x = np.random.rand(30)
        y = np.random.rand(30)
        ax.scatter(x, y)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title(f'Diagram {index}: Scatter Plot')
        
    else:
        # Default to a line graph
        x = np.linspace(0, 10, 100)
        
        if "sine" in desc_lower or "sin" in desc_lower:
            y = np.sin(x)
            title = "Sine Wave"
        elif "cosine" in desc_lower or "cos" in desc_lower:
            y = np.cos(x)
            title = "Cosine Wave"
        elif "exponential" in desc_lower or "exp" in desc_lower:
            y = np.exp(x/5) / np.exp(2)  # Scaled exponential
            title = "Exponential Function"
        elif "logarithm" in desc_lower or "log" in desc_lower:
            y = np.log(x + 1)
            title = "Logarithmic Function"
        elif "parabola" in desc_lower or "quadratic" in desc_lower:
            y = x**2 / 10
            title = "Quadratic Function"
        else:
            # Generate a simple line
            y = x / 2
            title = "Linear Function"
        
        ax.plot(x, y)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title(f'Diagram {index}: {title}')
        ax.grid(True)
    
    # Add a border
    plt.tight_layout()
    
    # Save to BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_circuit_diagram(description, index, width=600, height=400):
    """Create a simple circuit diagram based on the description"""
    # Create a blank image with a white background
    img = PILImage.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw a border
    draw.rectangle([(10, 10), (width-10, height-10)], outline='black', width=2)
    
    # Add a title
    draw.text((width//2-120, 20), f"Circuit Diagram {index}", fill='black', font=title_font)
    
    # Draw a simple circuit based on description
    desc_lower = description.lower()
    
    # Base coordinates for the circuit
    left_x = 100
    right_x = width - 100
    top_y = 100
    bottom_y = height - 100
    
    # Draw the basic circuit loop
    draw.line([(left_x, top_y), (right_x, top_y)], fill='black', width=3)
    draw.line([(right_x, top_y), (right_x, bottom_y)], fill='black', width=3)
    draw.line([(right_x, bottom_y), (left_x, bottom_y)], fill='black', width=3)
    draw.line([(left_x, bottom_y), (left_x, top_y)], fill='black', width=3)
    
    # Add battery symbol
    if "battery" in desc_lower or "cell" in desc_lower:
        # Draw battery on the left side
        battery_x = left_x
        battery_top = top_y + 50
        battery_bottom = battery_top + 80
        
        # Positive terminal
        draw.line([(battery_x-15, battery_top), (battery_x+15, battery_top)], fill='black', width=3)
        draw.line([(battery_x, battery_top-10), (battery_x, battery_top+10)], fill='black', width=3)
        
        # Negative terminal
        draw.line([(battery_x-15, battery_top+40), (battery_x+15, battery_top+40)], fill='black', width=3)
        
        # Label
        draw.text((battery_x-30, battery_top+20), "Battery", fill='black', font=font)
    
    # Add resistor symbol
    if "resistor" in desc_lower:
        # Draw resistor on the top
        resistor_y = top_y
        resistor_left = left_x + 100
        resistor_right = resistor_left + 100
        
        # Zigzag resistor symbol
        points = []
        x = resistor_left
        zigzag_height = 15
        while x < resistor_right:
            y_offset = zigzag_height if (x - resistor_left) % 20 < 10 else -zigzag_height
            points.append((x, resistor_y + y_offset))
            x += 10
        
        # Connect points with lines
        last_point = (resistor_left, resistor_y)
        for point in points:
            draw.line([last_point, point], fill='black', width=3)
            last_point = point
        draw.line([last_point, (resistor_right, resistor_y)], fill='black', width=3)
        
        # Label
        draw.text((resistor_left+30, resistor_y-40), "Resistor", fill='black', font=font)
    
    # Add bulb/lamp symbol
    if "bulb" in desc_lower or "lamp" in desc_lower:
        # Draw bulb on the right side
        bulb_x = right_x
        bulb_y = bottom_y - 80
        
        # Circle for bulb
        draw.ellipse([(bulb_x-25, bulb_y-25), (bulb_x+25, bulb_y+25)], outline='black', width=3)
        
        # Filament
        draw.line([(bulb_x-15, bulb_y), (bulb_x+15, bulb_y)], fill='black', width=2)
        
        # X cross inside to represent filament
        draw.line([(bulb_x-15, bulb_y-15), (bulb_x+15, bulb_y+15)], fill='black', width=2)
        draw.line([(bulb_x-15, bulb_y+15), (bulb_x+15, bulb_y-15)], fill='black', width=2)
        
        # Label
        draw.text((bulb_x+30, bulb_y), "Lamp", fill='black', font=font)
    
    # Add switch symbol
    if "switch" in desc_lower:
        # Draw switch on the bottom
        switch_y = bottom_y
        switch_left = left_x + 150
        switch_right = switch_left + 80
        
        # Switch symbol
        draw.line([(switch_left, switch_y), (switch_left+20, switch_y)], fill='black', width=3)
        draw.line([(switch_right-20, switch_y), (switch_right, switch_y)], fill='black', width=3)
        
        # Switch lever (open position)
        draw.line([(switch_left+20, switch_y), (switch_right-30, switch_y-30)], fill='black', width=3)
        
        # Label
        draw.text((switch_left+20, switch_y+10), "Switch", fill='black', font=font)
    
    # Save to BytesIO
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def generate_geometric_diagram(description, index, width=600, height=400):
    """Create a geometric diagram based on the description"""
    # Create a figure
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    # Determine the type of geometric shape from the description
    desc_lower = description.lower()
    
    # Set plot limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Triangle
    if "triangle" in desc_lower:
        # Check for specific triangle types
        if "equilateral" in desc_lower:
            # Equilateral triangle
            x = [5, 3, 7]
            y = [8, 4, 4]
            triangle_type = "Equilateral Triangle"
        elif "isosceles" in desc_lower:
            # Isosceles triangle
            x = [5, 3, 7]
            y = [8, 4, 4]
            triangle_type = "Isosceles Triangle"
        elif "right" in desc_lower:
            # Right-angled triangle
            x = [2, 2, 7]
            y = [2, 7, 2]
            triangle_type = "Right-angled Triangle"
            
            # Add the right angle symbol
            ax.plot([2.5, 2.5], [2, 2.5], 'k-', linewidth=1)
            ax.plot([2, 2.5], [2.5, 2.5], 'k-', linewidth=1)
        else:
            # Generic triangle
            x = [2, 5, 8]
            y = [2, 8, 3]
            triangle_type = "Triangle"
        
        # Draw the triangle
        ax.fill(x, y, alpha=0.3)
        ax.plot(x + [x[0]], y + [y[0]], 'k-', linewidth=2)
        
        # Label vertices
        ax.text(x[0]-0.5, y[0]-0.5, 'A', fontsize=12)
        ax.text(x[1]-0.5, y[1]+0.5, 'B', fontsize=12)
        ax.text(x[2]+0.5, y[2]-0.5, 'C', fontsize=12)
        
        # Set title
        ax.set_title(f'Diagram {index}: {triangle_type}')
    
    # Square or Rectangle
    elif "square" in desc_lower or "rectangle" in desc_lower:
        if "square" in desc_lower:
            # Square
            x = [2, 2, 7, 7, 2]
            y = [2, 7, 7, 2, 2]
            shape_type = "Square"
        else:
            # Rectangle
            x = [2, 2, 8, 8, 2]
            y = [2, 6, 6, 2, 2]
            shape_type = "Rectangle"
        
        # Draw the shape
        ax.fill(x[:4], y[:4], alpha=0.3)
        ax.plot(x, y, 'k-', linewidth=2)
        
        # Label vertices
        ax.text(x[0]-0.5, y[0]-0.5, 'A', fontsize=12)
        ax.text(x[1]-0.5, y[1]+0.5, 'B', fontsize=12)
        ax.text(x[2]+0.5, y[2]+0.5, 'C', fontsize=12)
        ax.text(x[3]+0.5, y[3]-0.5, 'D', fontsize=12)
        
        # Set title
        ax.set_title(f'Diagram {index}: {shape_type}')
    
    # Circle
    elif "circle" in desc_lower:
        # Draw a circle
        circle = plt.Circle((5, 5), 3, fill=False, linewidth=2)
        ax.add_artist(circle)
        
        # Add center point
        ax.plot(5, 5, 'ko', markersize=5)
        ax.text(5+0.3, 5+0.3, 'O', fontsize=12)
        
        # Add radius line
        ax.plot([5, 8], [5, 5], 'k-', linewidth=1)
        ax.text(6.5, 5.3, 'r', fontsize=12)
        
        # Set title
        ax.set_title(f'Diagram {index}: Circle')
    
    # Angle
    elif "angle" in desc_lower:
        # Draw angle lines
        ax.plot([5, 9], [5, 5], 'k-', linewidth=2)  # Horizontal line
        
        # Determine angle from description
        if "30" in desc_lower:
            angle_deg = 30
        elif "45" in desc_lower:
            angle_deg = 45
        elif "60" in desc_lower:
            angle_deg = 60
        elif "90" in desc_lower:
            angle_deg = 90
        elif "120" in desc_lower:
            angle_deg = 120
        else:
            angle_deg = 45  # Default angle
            
        # Convert angle to radians
        angle_rad = math.radians(angle_deg)
        
        # Draw second line at the specified angle
        end_x = 5 + 4 * math.cos(angle_rad)
        end_y = 5 + 4 * math.sin(angle_rad)
        ax.plot([5, end_x], [5, end_y], 'k-', linewidth=2)
        
        # Draw arc to indicate angle
        angle_patch = plt.matplotlib.patches.Arc(
    xy=(5, 5),          # Center point
    width=2,            # Width of the ellipse
    height=2,           # Height of the ellipse
    angle=0,            # Rotation of the ellipse
    theta1=0,           # Starting angle in degrees
    theta2=angle_deg,   # Ending angle in degrees
    color='k',
    linewidth=1.5
)

        ax.add_patch(angle_patch)
        
        # Label the angle
        label_x = 5 + 0.7 * math.cos(angle_rad/2)
        label_y = 5 + 0.7 * math.sin(angle_rad/2)
        ax.text(label_x, label_y, f'{angle_deg}°', fontsize=12)
        
        # Set title
        ax.set_title(f'Diagram {index}: Angle {angle_deg}°')
    
    # Set equal aspect ratio and remove axes
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save to BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_biology_diagram(description, index, width=600, height=400):
    """Create a biology-related diagram based on the description"""
    # Create a blank image with a white background
    img = PILImage.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw a border
    draw.rectangle([(10, 10), (width-10, height-10)], outline='black', width=2)
    
    # Determine the type of biology diagram
    desc_lower = description.lower()
    
    # Add a title
    if "cell" in desc_lower:
        title = f"Cell Diagram {index}"
        draw.text((width//2-100, 20), title, fill='black', font=title_font)
        
        # Determine if it's animal or plant cell
        if "plant" in desc_lower:
            # Draw plant cell (rectangular with cell wall)
            cell_x, cell_y = width//2, height//2
            cell_width, cell_height = 300, 200
            
            # Cell wall (outer rectangle)
            draw.rectangle([
                (cell_x - cell_width//2 - 10, cell_y - cell_height//2 - 10),
                (cell_x + cell_width//2 + 10, cell_y + cell_height//2 + 10)
            ], outline='green', width=3)
            
            # Cell membrane (inner rectangle)
            draw.rectangle([
                (cell_x - cell_width//2, cell_y - cell_height//2),
                (cell_x + cell_width//2, cell_y + cell_height//2)
            ], outline='black', width=2)
            
            # Nucleus
            nucleus_x, nucleus_y = cell_x - 50, cell_y
            draw.ellipse([
                (nucleus_x - 30, nucleus_y - 25),
                (nucleus_x + 30, nucleus_y + 25)
            ], outline='black', width=2)
            draw.text((nucleus_x - 25, nucleus_y - 10), "Nucleus", fill='black', font=small_font)
            
            # Chloroplast (green ovals)
            for i in range(5):
                cp_x = cell_x + random.randint(-100, 100)
                cp_y = cell_y + random.randint(-70, 70)
                if i == 0:  # Label only one chloroplast
                    draw.ellipse([
                        (cp_x - 20, cp_y - 10),
                        (cp_x + 20, cp_y + 10)
                    ], fill='lightgreen', outline='green', width=1)
                    draw.text((cp_x - 15, cp_y + 15), "Chloroplast", fill='green', font=small_font)
                else:
                    draw.ellipse([
                        (cp_x - 20, cp_y - 10),
                        (cp_x + 20, cp_y + 10)
                    ], fill='lightgreen', outline='green', width=1)
            
            # Central vacuole
            vac_x, vac_y = cell_x + 50, cell_y
            draw.ellipse([
                (vac_x - 50, vac_y - 40),
                (vac_x + 50, vac_y + 40)
            ], outline='blue', width=2)
            draw.text((vac_x - 30, vac_y), "Vacuole", fill='blue', font=small_font)
            
            # Cell wall label
            draw.text((cell_x - cell_width//2 - 60, cell_y), "Cell wall", fill='green', font=small_font)
            
        else:
            # Draw animal cell (circular)
            cell_x, cell_y = width//2, height//2
            cell_radius = 150
            
            # Cell membrane (circle)
            draw.ellipse([
                (cell_x - cell_radius, cell_y - cell_radius),
                (cell_x + cell_radius, cell_y + cell_radius)
            ], outline='black', width=2)
            
            # Nucleus
            nucleus_x, nucleus_y = cell_x - 30, cell_y
            draw.ellipse([
                (nucleus_x - 30, nucleus_y - 25),
                (nucleus_x + 30, nucleus_y + 25)
            ], outline='black', width=2)
            draw.text((nucleus_x - 25, nucleus_y - 10), "Nucleus", fill='black', font=small_font)
            
            # Mitochondria (bean-shaped)
            mito_x, mito_y = cell_x + 50, cell_y - 40
            # Draw a bean-shaped mitochondrion
            draw.arc([
                (mito_x - 25, mito_y - 15),
                (mito_x + 25, mito_y + 15)
            ], 0, 180, fill='red', width=2)
            draw.arc([
                (mito_x - 25, mito_y - 5),
                (mito_x + 25, mito_y + 25)
            ], 180, 360, fill='red', width=2)
            draw.text((mito_x - 15, mito_y + 25), "Mitochondrion", fill='red', font=small_font)
            
            # Endoplasmic Reticulum
            er_x, er_y = cell_x - 70, cell_y + 50
            for i in range(4):
                draw.line([
                    (er_x - 40, er_y - 10 + i*8),
                    (er_x + 40, er_y - 10 + i*8)
                ], fill='purple', width=2)
            draw.text((er_x - 40, er_y + 30), "Endoplasmic Reticulum", fill='purple', font=small_font)
            
    elif "organ" in desc_lower:
        if "heart" in desc_lower:
            title = f"Heart Diagram {index}"
            draw.text((width//2-100, 20), title, fill='black', font=title_font)
            
            # Draw heart outline (simplified)
            heart_x, heart_y = width//2, height//2
            
            # Heart shape
            # Left lobe
            draw.arc([
                (heart_x - 100, heart_y - 100),
                (heart_x, heart_y)
            ], 180, 0, fill='red', width=3)
            
            # Right lobe
            draw.arc([
                (heart_x, heart_y - 100),
                (heart_x + 100, heart_y)
            ], 180, 0, fill='red', width=3)
            
            # Bottom point
            draw.polygon([
                (heart_x - 100, heart_y - 50),
                (heart_x + 100, heart_y - 50),
                (heart_x, heart_y + 100)
            ], outline='red', width=3)
            
            # Label chambers
            draw.text((heart_x - 70, heart_y - 80), "Left Atrium", fill='black', font=small_font)
            draw.text((heart_x + 20, heart_y - 80), "Right Atrium", fill='black', font=small_font)
            draw.text((heart_x - 70, heart_y + 20), "Left Ventricle", fill='black', font=small_font)
            draw.text((heart_x + 20, heart_y + 20), "Right Ventricle", fill='black', font=small_font)
            
            # Main blood vessels
            draw.line([(heart_x, heart_y - 120), (heart_x, heart_y - 180)], fill='blue', width=4)
            draw.text((heart_x + 10, heart_y - 150), "Aorta", fill='blue', font=small_font)
            
        elif "brain" in desc_lower:
            title = f"Brain Diagram {index}"
            draw.text((width//2-100, 20), title, fill='black', font=title_font)
            
            # Draw brain outline
            brain_x, brain_y = width//2, height//2
            
            # Brain shape (simplified)
            draw.ellipse([
                (brain_x - 120, brain_y - 80),
                (brain_x + 120, brain_y + 100)
            ], outline='gray', width=3)
            
            # Cerebrum division
            draw.line([
                (brain_x, brain_y - 80),
                (brain_x, brain_y + 50)
            ], fill='black', width=2)
            
            # Cerebellum
            draw.ellipse([
                (brain_x - 60, brain_y + 60),
                (brain_x + 60, brain_y + 120)
            ], outline='gray', width=2)
            
            # Labels
            draw.text((brain_x - 100, brain_y - 50), "Left Hemisphere", fill='black', font=small_font)
            draw.text((brain_x + 10, brain_y - 50), "Right Hemisphere", fill='black', font=small_font)
            draw.text((brain_x - 30, brain_y + 80), "Cerebellum", fill='black', font=small_font)
            draw.text((brain_x - 70, brain_y + 20), "Frontal Lobe", fill='black', font=small_font)
            
        else:
            # Generic organ
            title = f"Organ Diagram {index}"
            draw.text((width//2-100, 20), title, fill='black', font=title_font)
            
            # Draw generic blob shape
            organ_x, organ_y = width//2, height//2
            
            # Blob shape
            points = []
            for angle in range(0, 360, 20):
                rad = math.radians(angle)
                radius = 100 + random.randint(-20, 20)
                x = organ_x + int(radius * math.cos(rad))
                y = organ_y + int(radius * math.sin(rad))
                points.append((x, y))
            
            # Connect the points
            for i in range(len(points) - 1):
                draw.line([points[i], points[i+1]], fill='brown', width=3)
            draw.line([points[-1], points[0]], fill='brown', width=3)
            
            # Add a label in the center
            draw.text((organ_x - 40, organ_y), "Organ Structure", fill='black', font=font)
            
    elif "plant" in desc_lower:
        title = f"Plant Diagram {index}"
        draw.text((width//2-100, 20), title, fill='black', font=title_font)
        
        # Draw a simple plant
        plant_x, plant_y = width//2, height - 100
        
        # Stem
        draw.line([
            (plant_x, plant_y),
            (plant_x, plant_y - 200)
        ], fill='green', width=5)
        
        # Roots
        for i in range(5):
            angle = 30 + i * 30
            length = 30 + random.randint(0, 30)
            end_x = plant_x + int(length * math.cos(math.radians(angle)))
            end_y = plant_y + int(length * math.sin(math.radians(angle)))
            draw.line([(plant_x, plant_y), (end_x, end_y)], fill='brown', width=2)
        
        # Leaves
        for i in range(3):
            y_pos = plant_y - 80 - i * 60
            # Left leaf
            draw.ellipse([
                (plant_x - 80, y_pos - 20),
                (plant_x, y_pos + 20)
            ], outline='green', width=2)
            # Right leaf
            draw.ellipse([
                (plant_x, y_pos - 20),
                (plant_x + 80, y_pos + 20)
            ], outline='green', width=2)
        
        # Flower
        flower_y = plant_y - 220
        # Petals
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            petal_x1 = plant_x + int(20 * math.cos(rad))
            petal_y1 = flower_y + int(20 * math.sin(rad))
            petal_x2 = plant_x + int(40 * math.cos(rad))
            petal_y2 = flower_y + int(40 * math.sin(rad))
            draw.ellipse([
                (petal_x1 - 10, petal_y1 - 10),
                (petal_x2 + 10, petal_y2 + 10)
            ], fill='yellow', outline='orange', width=1)
        
        # Center of flower
        draw.ellipse([
            (plant_x - 15, flower_y - 15),
            (plant_x + 15, flower_y + 15)
        ], fill='orange', outline='orange', width=1)
        
        # Labels
        draw.text((plant_x + 10, plant_y - 150), "Stem", fill='black', font=small_font)
        draw.text((plant_x + 10, plant_y + 20), "Roots", fill='black', font=small_font)
        draw.text((plant_x + 50, plant_y - 100), "Leaf", fill='black', font=small_font)
        draw.text((plant_x + 10, flower_y - 40), "Flower", fill='black', font=small_font)
    
    else:
        # Generic biology diagram
        title = f"Biology Diagram {index}"
        draw.text((width//2-100, 20), title, fill='black', font=title_font)
        draw.text((width//2-150, height//2), description, fill='black', font=font)
    
    # Save to BytesIO
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def generate_chemistry_diagram(description, index, width=600, height=400):
    """Create a chemistry-related diagram based on the description"""
    # Create a blank image with a white background
    img = PILImage.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw a border
    draw.rectangle([(10, 10), (width-10, height-10)], outline='black', width=2)
    
    # Determine the type of chemistry diagram
    desc_lower = description.lower()
    
    # Add a title
    if "atom" in desc_lower:
        element = None
        # Try to identify the element
        elements = [
            "hydrogen", "helium", "lithium", "beryllium", "boron", "carbon",
            "nitrogen", "oxygen", "fluorine", "neon", "sodium", "magnesium"
        ]
        for e in elements:
            if e in desc_lower:
                element = e.capitalize()
                break
        
        if not element:
            element = "Carbon"  # Default element
        
        title = f"{element} Atom Diagram {index}"
        draw.text((width//2-120, 20), title, fill='black', font=title_font)
        
        # Draw Bohr model
        atom_x, atom_y = width//2, height//2
        
        # Element configurations (simplified)
        electron_configs = {
            "Hydrogen": [1],
            "Helium": [2],
            "Lithium": [2, 1],
            "Beryllium": [2, 2],
            "Boron": [2, 3],
            "Carbon": [2, 4],
            "Nitrogen": [2, 5],
            "Oxygen": [2, 6],
            "Fluorine": [2, 7],
            "Neon": [2, 8],
            "Sodium": [2, 8, 1],
            "Magnesium": [2, 8, 2]
        }
        
        config = electron_configs.get(element, [2, 4])  # Default to carbon if not found
        
        # Nucleus
        nucleus_radius = 25
        draw.ellipse([
            (atom_x - nucleus_radius, atom_y - nucleus_radius),
            (atom_x + nucleus_radius, atom_y + nucleus_radius)
        ], fill='red', outline='black', width=2)
        
        draw.text((atom_x - 20, atom_y - 10), element, fill='white', font=font)
        
        # Electron shells
        shell_radii = [70, 120, 170]
        
        for i, shell_radius in enumerate(shell_radii[:len(config)]):
            # Draw the shell orbit
            draw.ellipse([
                (atom_x - shell_radius, atom_y - shell_radius),
                (atom_x + shell_radius, atom_y + shell_radius)
            ], outline='blue', width=2)
            
            # Add electrons
            electrons_in_shell = config[i]
            for e in range(electrons_in_shell):
                angle = (e * 360 / electrons_in_shell) * (math.pi / 180)
                e_x = atom_x + int(shell_radius * math.cos(angle))
                e_y = atom_y + int(shell_radius * math.sin(angle))
                
                # Draw electron
                draw.ellipse([
                    (e_x - 5, e_y - 5),
                    (e_x + 5, e_y + 5)
                ], fill='blue', outline='black', width=1)
        
        # Label
        draw.text((width//2 - 110, height - 50), 
                 f"Electron configuration: {' - '.join(str(n) for n in config)}", 
                 fill='black', font=font)
        
    elif "molecule" in desc_lower or "compound" in desc_lower:
        # Try to identify the molecule
        molecules = {
            "water": "H₂O",
            "carbon dioxide": "CO₂",
            "methane": "CH₄",
            "glucose": "C₆H₁₂O₆",
            "ammonia": "NH₃",
            "oxygen": "O₂"
        }
        
        molecule_name = None
        formula = None
        
        for name, chem_formula in molecules.items():
            if name in desc_lower:
                molecule_name = name.capitalize()
                formula = chem_formula
                break
        
        if not molecule_name:
            molecule_name = "Water"
            formula = "H₂O"
        
        title = f"{molecule_name} ({formula}) Molecule Diagram {index}"
        draw.text((width//2-150, 20), title, fill='black', font=title_font)
        
        # Draw specific molecules
        mol_x, mol_y = width//2, height//2
        
        if "water" in desc_lower or (not molecule_name and formula == "H₂O"):
            # Draw water molecule (H2O)
            # Oxygen atom
            draw.ellipse([
                (mol_x - 25, mol_y - 25),
                (mol_x + 25, mol_y + 25)
            ], fill='red', outline='black', width=2)
            draw.text((mol_x - 10, mol_y - 10), "O", fill='white', font=font)
            
            # Hydrogen atoms
            h1_x, h1_y = mol_x - 60, mol_y - 20
            draw.ellipse([
                (h1_x - 15, h1_y - 15),
                (h1_x + 15, h1_y + 15)
            ], fill='lightblue', outline='black', width=2)
            draw.text((h1_x - 5, h1_y - 5), "H", fill='black', font=font)
            
            h2_x, h2_y = mol_x - 60, mol_y + 20
            draw.ellipse([
                (h2_x - 15, h2_y - 15),
                (h2_x + 15, h2_y + 15)
            ], fill='lightblue', outline='black', width=2)
            draw.text((h2_x - 5, h2_y - 5), "H", fill='black', font=font)
            
            # Bonds
            draw.line([(mol_x - 25, mol_y - 10), (h1_x + 15, h1_y)], fill='black', width=2)
            draw.line([(mol_x - 25, mol_y + 10), (h2_x + 15, h2_y)], fill='black', width=2)
            
        elif "carbon dioxide" in desc_lower:
            # Draw CO2
            # Carbon atom
            draw.ellipse([
                (mol_x - 20, mol_y - 20),
                (mol_x + 20, mol_y + 20)
            ], fill='gray', outline='black', width=2)
            draw.text((mol_x - 5, mol_y - 5), "C", fill='white', font=font)
            
            # Oxygen atoms
            o1_x, o1_y = mol_x - 70, mol_y
            draw.ellipse([
                (o1_x - 20, o1_y - 20),
                (o1_x + 20, o1_y + 20)
            ], fill='red', outline='black', width=2)
            draw.text((o1_x - 5, o1_y - 5), "O", fill='white', font=font)
            
            o2_x, o2_y = mol_x + 70, mol_y
            draw.ellipse([
                (o2_x - 20, o2_y - 20),
                (o2_x + 20, o2_y + 20)
            ], fill='red', outline='black', width=2)
            draw.text((o2_x - 5, o2_y - 5), "O", fill='white', font=font)
            
            # Double bonds
            draw.line([(mol_x - 20, mol_y - 5), (o1_x + 20, o1_y - 5)], fill='black', width=2)
            draw.line([(mol_x - 20, mol_y + 5), (o1_x + 20, o1_y + 5)], fill='black', width=2)
            
            draw.line([(mol_x + 20, mol_y - 5), (o2_x - 20, o2_y - 5)], fill='black', width=2)
            draw.line([(mol_x + 20, mol_y + 5), (o2_x - 20, o2_y + 5)], fill='black', width=2)
            
        else:
            # Generic molecule representation
            # Central atom
            draw.ellipse([
                (mol_x - 30, mol_y - 30),
                (mol_x + 30, mol_y + 30)
            ], fill='gray', outline='black', width=2)
            
            # Surrounding atoms in a tetrahedral arrangement
            surrounding_positions = [
                (mol_x, mol_y - 80),  # top
                (mol_x - 70, mol_y + 40),  # bottom left
                (mol_x + 70, mol_y + 40),  # bottom right
                (mol_x, mol_y + 80)   # closer to viewer
            ]
            
            for i, pos in enumerate(surrounding_positions):
                s_x, s_y = pos
                # Different colors for different atoms
                if i % 3 == 0:
                    color = 'red'  # oxygen
                    label = "O"
                elif i % 3 == 1:
                    color = 'blue'  # nitrogen
                    label = "N"
                else:
                    color = 'lightblue'  # hydrogen
                    label = "H"
                    
                draw.ellipse([
                    (s_x - 20, s_y - 20),
                    (s_x + 20, s_y + 20)
                ], fill=color, outline='black', width=2)
                draw.text((s_x - 5, s_y - 5), label, fill='white', font=font)
                
                # Bond
                draw.line([(mol_x, mol_y), (s_x, s_y)], fill='black', width=2)
            
        # Add chemical formula at the bottom
        draw.text((width//2 - 50, height - 50), formula, fill='black', font=title_font)
        
    elif "reaction" in desc_lower:
        title = f"Chemical Reaction Diagram {index}"
        draw.text((width//2-150, 20), title, fill='black', font=title_font)
        
        # Determine reaction type
        reaction_type = "generic"
        if "combustion" in desc_lower:
            reaction_type = "combustion"
        elif "acid" in desc_lower and "base" in desc_lower:
            reaction_type = "acid_base"
        elif "precipitation" in desc_lower:
            reaction_type = "precipitation"
            
        # Draw the reaction
        if reaction_type == "combustion":
            # Methane combustion: CH4 + 2O2 → CO2 + 2H2O
            # Reactants
            draw.text((100, 150), "CH₄", fill='black', font=title_font)
            draw.text((200, 150), "+", fill='black', font=title_font)
            draw.text((250, 150), "2O₂", fill='black', font=title_font)
            
            # Arrow
            draw.line([(350, 150), (450, 150)], fill='black', width=3)
            draw.polygon([(440, 140), (450, 150), (440, 160)], fill='black')
            
            # Products
            draw.text((470, 150), "CO₂", fill='black', font=title_font)
            draw.text((550, 150), "+", fill='black', font=title_font)
            draw.text((600, 150), "2H₂O", fill='black', font=title_font)
            
            # Add heat
            draw.text((380, 120), "heat", fill='red', font=font)
            
            # Add flames
            for i in range(5):
                x = 380 + i * 10
                draw.line([(x, 180), (x, 200)], fill='red', width=2)
                draw.line([(x, 180), (x-5, 170)], fill='orange', width=2)
                draw.line([(x, 180), (x+5, 170)], fill='orange', width=2)
                
        elif reaction_type == "acid_base":
            # HCl + NaOH → NaCl + H2O
            # Reactants
            draw.text((100, 150), "HCl", fill='red', font=title_font)
            draw.text((180, 150), "+", fill='black', font=title_font)
            draw.text((220, 150), "NaOH", fill='blue', font=title_font)
            
            # Arrow
            draw.line([(350, 150), (450, 150)], fill='black', width=3)
            draw.polygon([(440, 140), (450, 150), (440, 160)], fill='black')
            
            # Products
            draw.text((470, 150), "NaCl", fill='black', font=title_font)
            draw.text((550, 150), "+", fill='black', font=title_font)
            draw.text((600, 150), "H₂O", fill='black', font=title_font)
            
        elif reaction_type == "precipitation":
            # AgNO3 + NaCl → AgCl↓ + NaNO3
            # Reactants
            draw.text((100, 150), "AgNO₃", fill='black', font=title_font)
            draw.text((200, 150), "+", fill='black', font=title_font)
            draw.text((250, 150), "NaCl", fill='black', font=title_font)
            
            # Arrow
            draw.line([(350, 150), (450, 150)], fill='black', width=3)
            draw.polygon([(440, 140), (450, 150), (440, 160)], fill='black')
            
            # Products
            draw.text((470, 150), "AgCl↓", fill='brown', font=title_font)
            draw.text((550, 150), "+", fill='black', font=title_font)
            draw.text((600, 150), "NaNO₃", fill='black', font=title_font)
            
            # Draw precipitate
            for i in range(8):
                x = 500 + i * 5
                y = 220 + (i % 3) * 10
                draw.rectangle([(x-3, y-3), (x+3, y+3)], fill='brown')
                
        else:
            # Generic reaction: A + B → C + D
            # Reactants
            draw.text((150, 150), "A", fill='blue', font=title_font)
            draw.text((200, 150), "+", fill='black', font=title_font)
            draw.text((250, 150), "B", fill='red', font=title_font)
            
            # Arrow
            draw.line([(350, 150), (450, 150)], fill='black', width=3)
            draw.polygon([(440, 140), (450, 150), (440, 160)], fill='black')
            
            # Products
            draw.text((500, 150), "C", fill='green', font=title_font)
            draw.text((550, 150), "+", fill='black', font=title_font)
            draw.text((600, 150), "D", fill='purple', font=title_font)
            
        # Add reaction conditions
        draw.text((width//2 - 100, height - 50), 
                 "Reaction conditions: Standard temp & pressure", 
                 fill='black', font=font)
        
    else:
        # Generic chemistry diagram
        title = f"Chemistry Diagram {index}"
        draw.text((width//2-100, 20), title, fill='black', font=title_font)
        draw.text((width//2-150, height//2), description, fill='black', font=font)
    
    # Save to BytesIO
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


# Function to create PDF of generated questions
def create_pdf(questions):
    """
    Create a PDF document containing the generated questions
    """
    # Create a BytesIO buffer to store the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=10,
        spaceBefore=15
    )
    
    mark_scheme_style = ParagraphStyle(
        'MarkScheme',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        leftIndent=20,
        spaceBefore=5
    )
    
    difficulty_style = ParagraphStyle(
        'Difficulty',
        parent=styles['Normal'],
        fontSize=10,
        italic=True,
        spaceBefore=5
    )
    
    # List to hold content elements
    content = []
    
    # Add title
    title_text = "Generated Exam Questions"
    content.append(Paragraph(title_text, title_style))
    
    # Current date
    current_date = datetime.now().strftime("%B %d, %Y")
    content.append(Paragraph(f"Generated on: {current_date}", normal_style))
    content.append(Spacer(1, 20))
    
    # Add each question
    for i, q in enumerate(questions, 1):
        # Question number and text
        content.append(Paragraph(f"Question {i}", heading_style))
        
        # Add topic if available
        if 'topic' in q and q['topic']:
            content.append(Paragraph(f"<b>Topic:</b> {q['topic']}", normal_style))
        
        # Add difficulty if available
        if 'difficulty' in q and q['difficulty']:
            difficulty_color = {
                'Easy': 'green',
                'Medium': 'orange',
                'Hard': 'red'
            }.get(q['difficulty'], 'black')
            
            content.append(
                Paragraph(
                    f"<b>Difficulty:</b> <font color='{difficulty_color}'>{q['difficulty']}</font>", 
                    difficulty_style
                )
            )
        
        content.append(Spacer(1, 10))
        
        # Question text
        question_text = q['question'].replace('\n', '<br/>')
        content.append(Paragraph(question_text, normal_style))
        
        # Add diagrams if any
        if 'diagrams' in q and q['diagrams']:
            content.append(Spacer(1, 10))
            for j, diagram_data in enumerate(q['diagrams'], 1):
                img_io = diagram_data
                # Convert BytesIO to ReportLab Image
                img = Image(img_io, width=300, height=200)
                content.append(img)
                content.append(Paragraph(f"Diagram {j}", normal_style))
                content.append(Spacer(1, 10))
        
        # Add mark scheme
        if 'mark_scheme' in q and q['mark_scheme']:
            content.append(Paragraph("<b>Mark Scheme:</b>", heading_style))
            mark_scheme_text = q['mark_scheme'].replace('\n', '<br/>')
            content.append(Paragraph(mark_scheme_text, mark_scheme_style))
        
        # Add spacer between questions
        content.append(Spacer(1, 20))
    
    # Build the PDF
    doc.build(content)
    
    # Reset buffer position to start
    buffer.seek(0)
    return buffer


# Function to call Groq API to generate questions
def generate_questions_with_groq(subject, level, topics, num_questions, difficulty, question_type, model):
    """
    Generate questions using the Groq LLM API
    """
    # Set up headers with API key
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Prepare difficulty string
    difficulty_str = ""
    if difficulty != "Mixed":
        difficulty_str = f"All questions should be of {difficulty} difficulty level."
    else:
        difficulty_str = "Mix the difficulty levels with approximately equal numbers of Easy, Medium, and Hard questions."
    
    # Prepare question type string
    question_type_str = ""
    if question_type != "Mixed":
        question_type_str = f"All questions should be in the {question_type} format: {QUESTION_FORMATS[question_type]}"
    else:
        question_type_str = "Mix question types including multiple choice, short answer, calculation, and extended response formats."
    
    # Prepare topics string
    topics_str = ", ".join(topics)
    
    # Create the prompt
    prompt = f"""You are an exam question generator for {level} {subject}. Generate {num_questions} high-quality past paper style questions covering the following topics: {topics_str}.

{difficulty_str}
{question_type_str}

The questions should:
1. Match real {level} {subject} exam questions in style, format, and complexity
2. Include a detailed mark scheme showing how points are awarded
3. Be clearly labeled with their difficulty level (Easy, Medium, or Hard)
4. Include diagrams where appropriate - describe any needed diagrams in detail by enclosing the description in [DIAGRAM: description] tags

For each question, provide:
- The question itself
- The topic it covers
- The difficulty level
- A detailed mark scheme
- Any diagram descriptions in [DIAGRAM: description] format

Format your response as a JSON array of questions with the following structure:
```json
[
  {{
    "question": "The full text of the question...",
    "topic": "The specific topic",
    "difficulty": "Easy|Medium|Hard",
    "mark_scheme": "The full mark scheme...",
    "diagram_descriptions": ["Description 1", "Description 2"] // Only if diagrams are needed
  }},
  // More questions...
]
```

The generated questions should be challenging but fair, and should test understanding rather than just recall. Make the questions engaging and relevant to real-world applications where possible.
"""

    # Prepare the request payload
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "temperature": 0.7,
        "max_tokens": 4000,
        "top_p": 1,
        "stream": False
    }
    
    try:
        # Make the API request
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                               headers=headers, 
                               json=payload)
        
        # Check for successful response
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Extract the generated text
        generated_text = result['choices'][0]['message']['content']
        
        # Extract JSON from response
        # First, find the JSON part in the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', generated_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If not found in code blocks, try to extract anything that looks like JSON array
            json_match = re.search(r'\[\s*{[\s\S]*}\s*\]', generated_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = generated_text
        
        # Parse the JSON
        try:
            questions_data = json.loads(json_str)
            
            # Process diagrams for each question
            for question in questions_data:
                # Convert diagram descriptions to actual diagrams
                if 'diagram_descriptions' in question and question['diagram_descriptions']:
                    question['diagrams'] = []
                    for i, desc in enumerate(question['diagram_descriptions'], 1):
                        diagram_io = generate_diagram(desc, i)
                        question['diagrams'].append(diagram_io)
                    
                # Also check if there are diagram descriptions in the question text
                question_text, diagram_descs = process_diagram_text(question['question'])
                if diagram_descs:
                    if 'diagrams' not in question:
                        question['diagrams'] = []
                    
                    # Start index after any existing diagrams
                    start_idx = len(question.get('diagrams', [])) + 1
                    for i, desc in enumerate(diagram_descs, start_idx):
                        diagram_io = generate_diagram(desc, i)
                        question['diagrams'].append(diagram_io)
                    
                    # Update question text with cleaned version
                    question['question'] = question_text
            
            return questions_data
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {e}")
            st.text(json_str)
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"API request error: {e}")
        return []


# Main app layout
st.sidebar.markdown('<h2 class="sub-header">Exam Configuration</h2>', unsafe_allow_html=True)

# Select curriculum level
level = st.sidebar.selectbox("Select Curriculum Level", ["IGCSE", "A-Level"])

# Select subject based on level
available_subjects = SUBJECTS.get(level, [])
subject = st.sidebar.selectbox("Select Subject", available_subjects)

# Create a combined identifier for the subject
subject_key = f"{level} {subject}"

# Select topics based on subject and level
available_topics = TOPICS.get(subject_key, [])
if not available_topics and subject:
    st.sidebar.info(f"No specific topics available for {subject}. Questions will cover the general curriculum.")
    available_topics = ["General Curriculum"]

# If this is a new subject selection, reset the selected topics
if subject_key != st.session_state.selected_subject:
    st.session_state.selected_subject = subject_key
    st.session_state.selected_topics = []

selected_topics = st.sidebar.multiselect(
    "Select Topics", 
    available_topics,
    default=st.session_state.selected_topics
)

# Update session state with selected topics
st.session_state.selected_topics = selected_topics

# Number of questions
num_questions = st.sidebar.slider("Number of Questions", 1, 10, 3)

# Difficulty level
difficulty_options = ["Mixed", "Easy", "Medium", "Hard"]
difficulty = st.sidebar.selectbox("Difficulty Level", difficulty_options)

# Question format
question_format_options = ["Mixed"] + list(QUESTION_FORMATS.keys())
question_format = st.sidebar.selectbox("Question Format", question_format_options)

# Model selection
model = st.sidebar.selectbox("Select LLM Model", GROQ_MODELS, index=1)  # Default to llama3-70b

# API key input (optional - can use the pre-defined key)
custom_api_key = st.sidebar.text_input("Custom Groq API Key (optional)", type="password")
if custom_api_key:
    GROQ_API_KEY = custom_api_key

# Info box for API key
st.sidebar.markdown("""
<div class="info-box">
By default, the app uses a pre-configured API key. If you have your own Groq API key, you can enter it above for better reliability.
</div>
""", unsafe_allow_html=True)

# Generate button
generate_button = st.sidebar.button("Generate Questions", type="primary")

# Clear button
clear_button = st.sidebar.button("Clear Results")

# Download PDF button - only shown when questions are generated
if st.session_state.generated_questions:
    pdf_buffer = create_pdf(st.session_state.generated_questions)
    st.sidebar.download_button(
        label="Download as PDF",
        data=pdf_buffer,
        file_name=f"{level}_{subject}_questions.pdf",
        mime="application/pdf"
    )

# Clear results if requested
if clear_button:
    st.session_state.generated_questions = []
    st.experimental_rerun()

# Generate questions when the button is clicked
if generate_button:
    if not selected_topics:
        st.warning("Please select at least one topic.")
    else:
        with st.spinner(f"Generating {num_questions} questions for {level} {subject}..."):
            # Show a progress bar
            progress_bar = st.progress(0)
            
            # Generate the questions
            questions = generate_questions_with_groq(
                subject=subject,
                level=level,
                topics=selected_topics,
                num_questions=num_questions,
                difficulty=difficulty,
                question_type=question_format,
                model=model
            )
            
            # Update progress
            progress_bar.progress(100)
            
            # Store the generated questions in session state
            st.session_state.generated_questions = questions
            
            # Success message
            if questions:
                st.success(f"Successfully generated {len(questions)} questions!")
            else:
                st.error("Failed to generate questions. Please try again.")

# Display generated questions
if st.session_state.generated_questions:
    st.markdown('<h2 class="sub-header">Generated Questions</h2>', unsafe_allow_html=True)
    
    # Iterate through questions and display them
    for i, question in enumerate(st.session_state.generated_questions, 1):
        # Question box
        st.markdown(f"""
        <div class="question-box">
            <h3>Question {i}</h3>
            <p><strong>Topic:</strong> {question.get('topic', 'General')}</p>
            <p><strong>Difficulty:</strong> <span class="difficulty-{question.get('difficulty', 'Medium').lower()}">{question.get('difficulty', 'Medium')}</span></p>
            <p>{question['question'].replace(chr(10), '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display diagrams if any
        if 'diagrams' in question and question['diagrams']:
            st.markdown('<div class="diagram-box">', unsafe_allow_html=True)
            for j, diagram_data in enumerate(question['diagrams'], 1):
                st.image(diagram_data, caption=f"Diagram {j}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Mark scheme - initially hidden, with a button to show
        with st.expander("Show Mark Scheme"):
            st.markdown(f"""
            <div class="mark-scheme">
                {question.get('mark_scheme', 'Mark scheme not available').replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #6B7280; font-size: 0.8rem;">
    ExamPrep AI - Created for educational purposes | Powered by Groq LLM API
</div>
""", unsafe_allow_html=True)