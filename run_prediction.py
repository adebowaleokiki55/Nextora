import torch
from plasmosenet import PlasmoSENet
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# 1. Initialize the neural structure from your plasmosenet.py file
model = PlasmoSENet(num_classes=2)

# 2. Inject the weights you downloaded in model.pth
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 3. Setup the standard NIH image transformation format (matching your training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Open the full microscope field of view using OpenCV
try:
    image_path = "cell_image.png"
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        raise FileNotFoundError
        
    # Convert to grayscale and blur slightly to suppress fine background noise
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to segment dark cell boundaries from bright background
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 51, 7
    )
    
    # Find bounding outlines for individual cells
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    infected_count = 0
    total_cells = 0
    classes = ["Parasitized (Infected)", "Uninfected"]
    
    print("Processing individual cells detected in the slide...")
    
    # Loop over every detected object in the microscope view
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out tiny dust artifacts or massive overlapping clumps 
        # (Adjust these pixel thresholds based on your resolution if needed)
        if 40 < w < 250 and 40 < h < 250:
            total_cells += 1
            
            # Crop the specific single cell from the raw BGR image
            cell_crop = cv_img[y:y+h, x:x+w]
            
            # Convert OpenCV BGR array back to PIL RGB Image for PyTorch transforms
            cell_rgb = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2RGB)
            pil_cell = Image.fromarray(cell_rgb)
            
            # Formulate tensor matching the training profile
            input_tensor = transform(pil_cell).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                prediction = output.argmax(dim=1).item()
                
            if prediction == 0:  # "Parasitized (Infected)"
                infected_count += 1

    # 5. Output patient-level diagnostics
    if total_cells == 0:
        print("\n[Warning] No individual red blood cells could be detected. Check your threshold limits.")
    else:
        parasitemia_rate = (infected_count / total_cells) * 100
        print("\n" + "="*40)
        print(" DIAGNOSTIC ANALYSIS REPORT")
        print("="*40)
        print(f"Total Red Blood Cells Analyzed: {total_cells}")
        print(f"Infected Cells Detected:        {infected_count}")
        print(f"Calculated Parasitemia Rate:    {parasitemia_rate:.2f}%")
        print("-"*40)
        
        # Patient status based on calculated load
        if infected_count > 0:
            print("Final System Verdict:           MALARIA POSITIVE")
        else:
            print("Final System Verdict:           MALARIA NEGATIVE")
        print("="*40)

except FileNotFoundError:
    print("\n[Error] Drop a test blood smear image named 'cell_image.png' into this folder to test it!")
