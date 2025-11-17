# Quick Start Guide

## Get Up and Running in 5 Minutes!

### Step 1: Install Dependencies

```bash
# Navigate to project root
cd "/Users/wmuckelroy/Desktop/Fall 2025/image_processing/Justice-For-Short-Kings"

# Install all dependencies
pip install -r requirements.txt
```

This will install:
- PyTorch (for SAM)
- segment-anything (Meta's SAM)
- OpenCV (image processing)
- MediaPipe (person detection)
- And other utilities

**Note:** This might take 5-10 minutes depending on your internet connection.

### Step 2: Download SAM Model

```bash
# Download the base model (375 MB - fastest option)
python src/Will/download_sam_model.py
```

When prompted, choose option **1** (vit_b - Base model) for fastest results.

For better quality but slower processing, choose option **3** (vit_h - Huge model).

### Step 3: Run the Demo!

```bash
# Run with auto-generated test image
python src/Will/demo.py
```

This will:
1. Create a test image with stick figures of different heights
2. Segment all people using SAM
3. Expand the shortest person to match the tallest
4. Display a 4-panel visualization

### Step 4: Test with Your Own Image

```bash
python src/Will/demo.py path/to/your/photo.jpg
```

Best results with:
- Full-body shots
- People of different heights
- Clear backgrounds
- Good lighting

### Step 5: Use in Your Code

```python
from Will import SegmentationExpander
import cv2

# Initialize (do this once)
expander = SegmentationExpander(
    sam_checkpoint="src/Will/assets/sam_vit_b_01ec64.pth",
    sam_model_type="vit_b",
    device="cpu"
)

# Load and process image
image = cv2.imread("your_image.jpg")
results = expander.process_image(image)

# Display
expander.display_results(results)
```

## Common Issues

### "ModuleNotFoundError: No module named 'cv2'"

**Solution:** Run `pip install -r requirements.txt`

### "SAM checkpoint not found"

**Solution:** Run `python src/Will/download_sam_model.py`

### "CUDA out of memory"

**Solution:** Use CPU instead:
```python
expander = SegmentationExpander(device="cpu")
```

Or use smaller model (vit_b instead of vit_h)

### Slow Processing

**Solutions:**
1. Use GPU if available: `device="cuda"`
2. Use smaller model: `sam_model_type="vit_b"`
3. Resize images to smaller resolution before processing

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Integrate with the main GUI
- Test with video streams
- Experiment with different expansion methods (`stretch` vs `scale`)

---

Happy coding! If you have questions, check the README or ask Will.
