document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const uploadButton = document.getElementById("uploadButton");
  const fileInfo = document.getElementById("fileInfo");
  const fileNameText = document.getElementById("fileNameText");
  const fileSizeText = document.getElementById("fileSizeText");
  const progressBar = document.getElementById("progressBar");
  const uploadPercentText = document.getElementById("uploadPercentText");
  const imagePreview = document.getElementById("imagePreview");
  const downloadImageButton = document.getElementById("downloadImage"); // ✅ 다운로드 버튼

  let selectedFile = null;

  // 파일 사이즈 포맷터
  function formatBytes(bytes) {
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  // 파일 선택 처리
  function handleFileSelection(file) {
    selectedFile = file;
    fileNameText.textContent = selectedFile.name;
    fileSizeText.textContent = `0 MB of ${formatBytes(selectedFile.size)}`;
    uploadPercentText.textContent = 0;
    progressBar.style.width = "0%";
    progressBar.textContent = "0%";
    fileInfo.classList.remove("d-none");
    uploadButton.disabled = false;
  }

  // Drag & Drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("border-info");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("border-info");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-info");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelection(fileInput.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileSelection(fileInput.files[0]);
    }
  });

  // 업로드 버튼 클릭
  uploadButton.addEventListener("click", () => {
    if (!selectedFile) return;

    uploadButton.disabled = true;
    const formData = new FormData();
    formData.append("file", selectedFile);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://127.0.0.1:8001/upload", true);

    // 진행률 업데이트
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        progressBar.style.width = percent + "%";
        progressBar.textContent = percent + "%";
        uploadPercentText.textContent = percent;
        fileSizeText.textContent = `${formatBytes(e.loaded)} of ${formatBytes(e.total)}`;
      }
    });

    // 업로드 완료 처리
    xhr.onload = () => {
      console.log("📦 xhr status:", xhr.status);
      console.log("📦 xhr responseText:", xhr.responseText);
      if (xhr.status >= 200 && xhr.status < 300) {
        const result = JSON.parse(xhr.responseText);
        const gallery = document.getElementById("galleryContainer");
        gallery.innerHTML = "";

        const section4 = document.getElementById("section-4");
        const selectedImageContainer = document.getElementById(
          "selectedImageContainer"
        );

        function createFigure(imgSrc, titleText) {
          const fig = document.createElement("figure");
          fig.className = "effect-julia item";
          fig.innerHTML = `
                <img src="${imgSrc}" alt="${titleText}" style="cursor: pointer; max-width: 100%; height: auto;" />
              `;
          gallery.appendChild(fig);

          const img = fig.querySelector("img");
          if (img) {
            img.addEventListener("click", () => {
              console.log(`🖼️ ${titleText} 클릭됨`);
              selectedImageContainer.innerHTML = `
                    <img src="${imgSrc}" alt="${titleText}" class="img-fluid mb-3" style="max-width: 100%; border-radius: 10px;" />
                  `;
              section4.scrollIntoView({ behavior: "smooth" });
            });
          } else {
            console.warn("⚠️ 이미지가 fig에 없음");
          }
        }

        if (result.result_visualizing_LSTM) {
          createFigure(result.result_visualizing_LSTM, "모델 1 결과");
        }
        if (result.result_visualizing_LSTM_v2) {
          createFigure(result.result_visualizing_LSTM_v2, "모델 2 결과");
        }
        if (result.result_visualizing_LSTM_v2) {
          createFigure(result.result_visualizing_LSTM_v2, "모델 3 결과");
        }

        uploadButton.textContent = "DONE";
        uploadButton.classList.remove("btn-primary");
        uploadButton.classList.add("btn-success");
      } else {
        alert("업로드 실패: " + xhr.statusText);
      }
    };

    xhr.onerror = () => {
      alert("업로드 중 오류가 발생했습니다.");
      progressBar.style.width = "0%";
      progressBar.textContent = "0%";
      fileSizeText.textContent = "0 MB";
    };

    xhr.send(formData);
  });

  // ✅ 다운로드 버튼 클릭 시 Section4 이미지 저장
  downloadImageButton.addEventListener("click", () => {
    const img = document.querySelector("#selectedImageContainer img");
    if (!img) {
      alert("이미지를 먼저 선택해주세요.");
      return;
    }

    const link = document.createElement("a");
    link.href = img.src;
    const filename = img.alt.replace(/\s+/g, "_") + ".png";
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
});
