# **Regional Prompt Upscaler**

**Version:** 1.2  
**Author:** [Hallett Visual](https://hallett-ai.com)

The **Regional Prompt Upscaler** is an advanced upscaling extension for **Automatic1111 Web UI** or **Forge**. It enhances images by splitting them into tiles and applying region-specific prompts generated with Vision-Language Models (VLMs). This tool is perfect for adding unique details to complex scenes with minimal effort.

---

## **Features**

- **Four Prompt Generation Methods**:
  Generate prompts using advanced Vision-Language Models (BLIP, Florence-2, CLIP, ViT-GPT2) for precise image detection.
- **Auto Tile Scaling**:
  Dynamically adjusts tile size based on image dimensions for optimal results.
- **Regional Prompt Export to Excel**:
  Export regional prompts to an Excel file for easy editing and re-import them seamlessly.
- **Complex and Simple Prompt Options**:
  Choose between detailed or concise prompts depending on your needs.
- **Enhanced Feather and Overlap Controls**:
  Smoothly blend tiles with adjustable feathering and overlap percentages.
- **Quick LoRA Selection**:
  Select and apply LoRA models directly from a dropdown menu for streamlined workflows.
- **Broad Compatibility**:
  Works with Flux, Stable Diffusion (SD), and SDXL.

---

## **Installation**

### **Option 1: Automatic Installation (Recommended)**

1. **Download the repository** or get the ZIP file from the [Releases](../../releases) page.
2. Extract the contents to a folder on your computer.
3. Run the `install_free_upscaler.bat`:
   - The installer will detect your Automatic1111 or Forge root folder.
   - It will copy all necessary files and install the required dependencies automatically.
4. **Restart your WebUI**. The extension will appear in the "Extensions" tab.

---

### **Option 2: Manual Installation**

1. **Download the repository** or get the ZIP file from the [Releases](../../releases) page.
2. Extract the contents to a folder on your computer.
3. Copy the following files and folders to your WebUI installation:
   - Copy `__init__.py` and `requirements.txt` into:
     ```
     <Your WebUI Directory>/extensions/regional-prompt-upscaler-hallett/
     ```
   - Copy the contents of the `scripts/` folder into:
     ```
     <Your WebUI Directory>/extensions/regional-prompt-upscaler-hallett/scripts/
     ```
4. **Install dependencies**:
   - Open a terminal in your WebUI root folder and run:
     ```bash
     pip install -r extensions/regional-prompt-upscaler-hallett/requirements.txt
     ```
5. **Restart your WebUI**. The extension will appear in the "Extensions" tab.

---

## **Usage**

Once installed, follow these steps to use the Regional Prompt Upscaler:

1. **Load an Image**:
   Select an image in the **img2img** tab.
2. **Adjust Tile and Prompt Settings**:
   - Set tile size, feathering, and overlap.
   - Choose a prompt generation method (BLIP, CLIP, Florence-2, etc.).
   - Select "Simple" or "Complex" prompt generation.
   - Add optional LoRA models via the dropdown menu.
3. **Process the Image**:
   Click "Generate" to process the image region by region with the specified prompts.
4. **Optional: Export Prompts**:
   Export regional prompts to an Excel file for editing or reuse.

---

## **Requirements**

- **Python 3.8+**
- **Automatic1111 WebUI** or **Forge**
- Required Python libraries (installed via `requirements.txt`):
  - `torch`
  - `transformers`
  - `openpyxl`
  - `numpy`
  - `opencv-python`
  - `Pillow`
  - `gradio`
  - `clip-interrogator`

---

## **Example Workflow**

### **Input Image**
![Input Image](assets/input_example.jpg)

### **Upscaled Output**
![Upscaled Output](assets/output_example.jpg)

In this example, different prompts were applied to specific regions of the image, resulting in an intelligently upscaled and detailed output.

---

## **Changelog**

### **v1.2**
- Updated `install_free_upscaler.bat` for improved installation and file management.
- Added enhanced feathering and overlap controls.
- Improved compatibility with Flux and SDXL.
- Streamlined LoRA selection dropdown.
- Florence-2 detection for better prompts.

---

## **Contributing**

We welcome contributions! Feel free to submit issues, suggestions, or pull requests to improve the project.

---

## **License**

This project is licensed under the [GPLv3 License](LICENSE). You are free to use the software and modify it, but any derivative works must also be open-source and distributed under the same license.

---

## **Contact**

For inquiries, support, or custom features, visit [Hallett Visual](https://hallett-ai.com) or reach out via our contact page.
