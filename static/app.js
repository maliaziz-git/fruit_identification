/**
 * Upload, crop, and prediction flow for the fruit identification GUI.
 */

const fileInput = document.getElementById("file-input");
const fileInput2 = document.getElementById("file-input-2");
const cameraInput = document.getElementById("camera-input");
const cameraInput2 = document.getElementById("camera-input-2");
const takePhotoBtn = document.getElementById("take-photo-btn");
const takePhotoBtn2 = document.getElementById("take-photo-btn-2");
const dropZone = document.getElementById("drop-zone");
const dropZone2 = document.getElementById("drop-zone-2");
const browseBtn = document.getElementById("browse-btn");
const browseBtn2 = document.getElementById("browse-btn-2");
const identifyBtn = document.getElementById("identify-btn");
const identifyBtn2 = document.getElementById("identify-btn-2");
const uploadView = document.getElementById("upload-view");
const resultsView = document.getElementById("results-view");
const previewImg = document.getElementById("preview-img");
const loading = document.getElementById("loading");
const cropModal = document.getElementById("crop-modal");
const cropImage = document.getElementById("crop-image");
const cropHint = document.getElementById("crop-hint");
const editCropBtn = document.getElementById("edit-crop-btn");
const uploadPreviewWrap = document.getElementById("upload-preview-wrap");
const uploadPreviewImg = document.getElementById("upload-preview-img");
const uploadEditCropBtn = document.getElementById("upload-edit-crop-btn");
const downloadPosterBtn = document.getElementById("download-poster-btn");
const loadingMessage = document.getElementById("loading-message");

const MAX_BYTES = 8 * 1024 * 1024;
const ALLOWED_EXT = ["jpg", "jpeg", "png", "webp"];
const ALLOWED_MIME = ["image/jpeg", "image/png", "image/webp"];

let selectedFile = null;
let pendingFile = null;
let lastPrediction = null;
let cropper = null;
let previewObjectUrl = null;
let cropImageUrl = null;

function revokePreviewUrl() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
}

function validateFile(file) {
  if (!file) return false;
  const ext = file.name.includes(".")
    ? file.name.split(".").pop().toLowerCase()
    : "";
  const extOk = ALLOWED_EXT.includes(ext);
  const mimeOk = ALLOWED_MIME.includes(file.type);
  if (!extOk && !mimeOk) {
    alert("Please upload JPG, PNG, or WEBP.");
    return false;
  }
  if (file.size > MAX_BYTES) {
    alert("File must be 8 MB or smaller.");
    return false;
  }
  return true;
}

function updatePreviewImages(url) {
  previewImg.src = url;
  previewImg.classList.remove("hidden");
  if (uploadPreviewImg) uploadPreviewImg.src = url;
  if (uploadPreviewWrap) uploadPreviewWrap.classList.remove("hidden");
  if (dropZone) dropZone.classList.add("has-file");
  if (editCropBtn) editCropBtn.classList.remove("hidden");
}

function setPreview(file) {
  revokePreviewUrl();
  previewObjectUrl = URL.createObjectURL(file);
  updatePreviewImages(previewObjectUrl);
}

function hideUploadPreview() {
  if (uploadPreviewWrap) uploadPreviewWrap.classList.add("hidden");
  if (uploadPreviewImg) uploadPreviewImg.removeAttribute("src");
  if (dropZone) dropZone.classList.remove("has-file");
}

function destroyCropper() {
  if (cropper) {
    cropper.destroy();
    cropper = null;
  }
}

function closeCropModal() {
  destroyCropper();
  cropModal.classList.add("hidden");
  cropImage.removeAttribute("src");
  if (cropImageUrl) {
    URL.revokeObjectURL(cropImageUrl);
    cropImageUrl = null;
  }
  if (!selectedFile) {
    hideUploadPreview();
  }
  pendingFile = null;
}

function openCropModal(file) {
  pendingFile = file;
  cropHint.textContent = file.name;
  cropModal.classList.remove("hidden");

  cropImage.onload = function onCropImageLoad() {
    cropImage.onload = null;
    destroyCropper();
    cropper = new Cropper(cropImage, {
      viewMode: 1,
      dragMode: "crop",
      aspectRatio: NaN,
      autoCropArea: 0.85,
      responsive: true,
      background: false,
      guides: true,
      center: true,
      highlight: true,
      cropBoxMovable: true,
      cropBoxResizable: true,
      toggleDragModeOnDblclick: true,
    });
  };

  if (cropImageUrl) URL.revokeObjectURL(cropImageUrl);
  cropImageUrl = URL.createObjectURL(file);
  cropImage.src = cropImageUrl;
}

function acceptFile(file) {
  selectedFile = file;
  setPreview(file);
  identifyBtn.disabled = false;
  identifyBtn2.disabled = false;
}

function applyCrop() {
  if (!cropper || !pendingFile) return;

  const canvas = cropper.getCroppedCanvas({
    maxWidth: 1024,
    maxHeight: 1024,
    imageSmoothingEnabled: true,
    imageSmoothingQuality: "high",
    fillColor: "#ffffff",
  });

  if (!canvas) {
    alert("Could not crop this image. Try a different area.");
    return;
  }

  canvas.toBlob(
    (blob) => {
      if (!blob) {
        alert("Crop failed. Please try again.");
        return;
      }
      if (blob.size > MAX_BYTES) {
        alert("Cropped image exceeds 8 MB. Crop a smaller region.");
        return;
      }
      const base = pendingFile.name.replace(/\.[^.]+$/i, "") || "fruit";
      const cropped = new File([blob], base + "_cropped.jpg", {
        type: "image/jpeg",
      });
      acceptFile(cropped);
      closeCropModal();
    },
    "image/jpeg",
    0.92
  );
}

function useFullImage() {
  if (!pendingFile) return;
  acceptFile(pendingFile);
  closeCropModal();
}

function handleFile(file) {
  if (!validateFile(file)) return;
  setPreview(file);
  openCropModal(file);
}

function setupDropZone(zone, input) {
  zone.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    input.click();
  });
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) handleFile(input.files[0]);
    input.value = "";
  });
}

function setupCameraInput(input) {
  if (!input) return;
  input.addEventListener("change", () => {
    if (input.files.length) handleFile(input.files[0]);
    input.value = "";
  });
}

function setLoadingMessage(text) {
  if (loadingMessage) loadingMessage.textContent = text;
}

function resetUpload() {
  selectedFile = null;
  pendingFile = null;
  lastPrediction = null;
  fileInput.value = "";
  fileInput2.value = "";
  if (cameraInput) cameraInput.value = "";
  if (cameraInput2) cameraInput2.value = "";
  identifyBtn.disabled = true;
  identifyBtn2.disabled = true;
  if (downloadPosterBtn) downloadPosterBtn.disabled = true;
  revokePreviewUrl();
  previewImg.removeAttribute("src");
  previewImg.classList.add("hidden");
  hideUploadPreview();
  if (editCropBtn) editCropBtn.classList.add("hidden");
  closeCropModal();
  uploadView.classList.remove("hidden");
  resultsView.classList.add("hidden");
}

async function runPredict() {
  if (!selectedFile) {
    alert("Please choose and crop an image first.");
    return;
  }
  loading.classList.remove("hidden");
  setLoadingMessage("Identifying fruit...");
  const formData = new FormData();
  formData.append("image", selectedFile, selectedFile.name);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    uploadView.classList.add("hidden");
    resultsView.classList.remove("hidden");

    document.getElementById("result-name").textContent = data.prediction;
    document.getElementById("result-confidence").textContent =
      (data.confidence * 100).toFixed(2) + "%";

    const bars = document.getElementById("top-bars");
    bars.innerHTML = "";
    const colors = ["bar-high", "bar-mid", "bar-low"];
    data.top3.forEach((item, i) => {
      const pct = (item.confidence * 100).toFixed(2);
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML =
        '<span class="bar-name">' +
        item.class +
        "</span>" +
        '<div class="bar-track"><div class="bar-fill ' +
        colors[i] +
        '" style="width:' +
        pct +
        '%"></div></div>' +
        '<span class="bar-pct">' +
        pct +
        "%</span>";
      bars.appendChild(row);
    });

    const statusBox = document.getElementById("status-box");
    statusBox.className = "status-box";
    if (data.confidence >= 0.7) statusBox.classList.add("status-high");
    else if (data.confidence >= 0.4) statusBox.classList.add("status-mid");
    else statusBox.classList.add("status-low");
    document.getElementById("status-message").textContent = data.message;

    const info = data.fruit_info || {};
    document.getElementById("info-description").textContent =
      info.description || "—";
    const nutList = document.getElementById("info-nutrition");
    nutList.innerHTML = "";
    (info.nutrition || []).forEach((n) => {
      const li = document.createElement("li");
      li.textContent = n;
      nutList.appendChild(li);
    });
    document.getElementById("info-season").textContent = info.season || "—";
    document.getElementById("info-price").textContent = info.price || "—";
    document.getElementById("info-uses").textContent = info.uses || "—";

    lastPrediction = data;
    if (downloadPosterBtn) downloadPosterBtn.disabled = false;
  } catch (err) {
    alert(err.message);
  } finally {
    loading.classList.add("hidden");
    setLoadingMessage("Identifying fruit...");
  }
}

async function downloadPoster() {
  if (!selectedFile || !lastPrediction) {
    alert("Please identify a fruit first.");
    return;
  }

  loading.classList.remove("hidden");
  setLoadingMessage("Creating poster...");
  if (downloadPosterBtn) downloadPosterBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile, selectedFile.name);
  formData.append(
    "metadata",
    JSON.stringify({
      prediction: lastPrediction.prediction,
      confidence: lastPrediction.confidence,
      fruit_info: lastPrediction.fruit_info || {},
    })
  );
  try {
    const res = await fetch("/generate-poster", { method: "POST", body: formData });
    if (!res.ok) {
      let message = "Poster generation failed.";
      try {
        const data = await res.json();
        message = data.error || message;
      } catch (_) {
        /* response may not be JSON */
      }
      throw new Error(message);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const safeName = (lastPrediction.prediction || "fruit")
      .replace(/[^\w\s-]/g, "")
      .trim()
      .replace(/\s+/g, "-");
    const a = document.createElement("a");
    a.href = url;
    a.download = (safeName || "fruit") + "-poster.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(err.message);
  } finally {
    loading.classList.add("hidden");
    setLoadingMessage("Identifying fruit...");
    if (downloadPosterBtn) downloadPosterBtn.disabled = !lastPrediction;
  }
}

// --- Init ---
setupDropZone(dropZone, fileInput);
setupDropZone(dropZone2, fileInput2);
setupCameraInput(cameraInput);
setupCameraInput(cameraInput2);
browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});
browseBtn2.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput2.click();
});
if (takePhotoBtn && cameraInput) {
  takePhotoBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    cameraInput.click();
  });
}
if (takePhotoBtn2 && cameraInput2) {
  takePhotoBtn2.addEventListener("click", (e) => {
    e.stopPropagation();
    cameraInput2.click();
  });
}

identifyBtn.addEventListener("click", runPredict);
identifyBtn2.addEventListener("click", runPredict);
if (downloadPosterBtn) {
  downloadPosterBtn.addEventListener("click", downloadPoster);
}
document
  .getElementById("upload-another-btn")
  .addEventListener("click", resetUpload);

document.getElementById("crop-apply-btn").addEventListener("click", applyCrop);
document.getElementById("crop-full-btn").addEventListener("click", useFullImage);
document.getElementById("crop-cancel-btn").addEventListener("click", closeCropModal);
document
  .getElementById("crop-modal-backdrop")
  .addEventListener("click", closeCropModal);

function openCropForSelected() {
  const file = selectedFile || pendingFile;
  if (file) openCropModal(file);
}

if (editCropBtn) {
  editCropBtn.addEventListener("click", openCropForSelected);
}
if (uploadEditCropBtn) {
  uploadEditCropBtn.addEventListener("click", openCropForSelected);
}

cropModal.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeCropModal();
});
