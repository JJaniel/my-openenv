from PIL import Image, ImageDraw, ImageFont
import os

def create_diagram(output_path):
    # Image size and background
    width, height = 1200, 1000
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Colors
    colors = {
        'BLUE': (232, 240, 254),      # Layer 1
        'BLUE_BORDER': (66, 133, 244),
        'GREEN': (230, 244, 234),     # Layer 2
        'GREEN_BORDER': (52, 168, 83),
        'YELLOW': (254, 247, 224),    # Layer 3
        'YELLOW_BORDER': (251, 188, 4),
        'RED': (252, 232, 230),       # Layer 4
        'RED_BORDER': (234, 67, 53),
        'PURPLE': (243, 231, 245),    # Layer 5
        'PURPLE_BORDER': (156, 39, 176),
        'TEXT': (33, 33, 33),
        'ARROW': (100, 100, 100)
    }

    # Helper function to draw a layer box
    def draw_layer(x, y, w, h, bg, border, title, details):
        draw.rectangle([x, y, x + w, y + h], fill=bg, outline=border, width=3)
        draw.text((x + 20, y + 15), title, fill=border)
        # Detail text
        for idx, line in enumerate(details):
            draw.text((x + 20, y + 45 + (idx * 25)), line, fill=colors['TEXT'])

    # Helper function for arrows
    def draw_arrow(x, y1, y2):
        draw.line([x, y1, x, y2], fill=colors['ARROW'], width=4)
        draw.polygon([(x-10, y2-15), (x+10, y2-15), (x, y2)], fill=colors['ARROW'])

    # Main Title
    draw.text((width//2 - 200, 20), "Meta-PyTorch Hackathon: OpenEnv Architecture", fill=(0,0,0))

    # Layers
    layer_w, layer_h = 800, 140
    start_x = 200
    
    # Layer 1: External Agent
    draw_layer(start_x, 80, layer_w, layer_h, colors['BLUE'], colors['BLUE_BORDER'], 
               "LAYER 1: THE EXTERNAL AGENT (THE AI MODEL)", 
               ["[ Meta / Hugging Face Judge Agent ]", "Sends HTTP POST requests to the environment server"])
    
    draw_arrow(start_x + layer_w//2, 220, 250)

    # Layer 2: Interface
    draw_layer(start_x, 260, layer_w, layer_h, colors['GREEN'], colors['GREEN_BORDER'], 
               "LAYER 2: THE INTERFACE (FASTAPI SERVER) - [server/app.py]", 
               ["/reset (Starts episode), /step (Takes action), /state (Current progress)", 
                "Converts JSON to Python Pydantic objects"])

    draw_arrow(start_x + layer_w//2, 400, 430)

    # Layer 3: Engine
    draw_layer(start_x, 440, layer_w, layer_h, colors['YELLOW'], colors['YELLOW_BORDER'], 
               "LAYER 3: THE ENGINE (RL LOGIC) - [server/environment.py]", 
               ["TASK SELECTOR: Easy (Categorize), Medium (Prioritize), Hard (Extract)", 
                "GRADER: Normalizes reward calculation (0.0 to 1.0)"])

    draw_arrow(start_x + layer_w//2, 580, 610)

    # Layer 4: Data Models
    draw_layer(start_x, 620, layer_w, layer_h, colors['RED'], colors['RED_BORDER'], 
               "LAYER 4: THE DATA MODELS (PYDANTIC) - [models.py]", 
               ["EmailAction: {cat_id, priority, info, reasoning}", 
                "EmailObservation: {subject, body, reward, done}"])

    draw_arrow(start_x + layer_w//2, 760, 790)

    # Layer 5: Reporter
    draw_layer(start_x, 800, layer_w, layer_h, colors['PURPLE'], colors['PURPLE_BORDER'], 
               "LAYER 5: THE REPORTER (LOGGING) - [inference.py]", 
               ["Mandatory logs for judge system scoring", 
                "[START] task=... [STEP] reward=... [END] score=..."])

    # Save image
    img.save(output_path)
    print(f"✅ Image saved to: {output_path}")

if __name__ == "__main__":
    create_diagram("My learning/architecture_diagram.png")
