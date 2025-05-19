document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const uploadButton = document.getElementById("uploadButton");
  const fileInfo = document.getElementById("fileInfo");
  const fileNameText = document.getElementById("fileNameText");
  const fileSizeText = document.getElementById("fileSizeText");
  const progressBar = document.getElementById("progressBar");
  const uploadPercentText = document.getElementById("uploadPercentText");
  const gallery = document.getElementById("galleryContainer");
  const selectedImageContainer = document.getElementById("selectedImageContainer");
  const downloadImageButton = document.getElementById("downloadImage");
  const report = document.getElementById('reportContent');
  let selectedFile = null;

  // 파일 크기 단위 변환 함수
  function formatBytes(bytes) {
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  // 파일 선택 시 처리
  function handleFileSelection(file) {
    selectedFile = file;
    fileNameText.textContent = selectedFile.name;
    fileSizeText.textContent = `0 MB of ${formatBytes(selectedFile.size)}`;
    uploadPercentText.textContent = "0";
    progressBar.style.width = "0%";
    progressBar.textContent = "0%";
    fileInfo.classList.remove("d-none");
    uploadButton.disabled = false;
  }

  // 드래그앤드롭 이벤트
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

  // 파일 input 변경 이벤트
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileSelection(fileInput.files[0]);
    }
  });

  // 서버 응답 결과 처리 함수
  function displayResults(result) {
    //보고서 생성
    report.innerText = result["보고서"] || "보고서가 없습니다.";
    // 테이블에 예측값, 재학습여부 업데이트
    const fishMap = {
      광어: { amountId: "flounder-amount", retrainId: "flounder-model-isRetrained" },
      연어: { amountId: "salmon-amount", retrainId: "salmon-model-isRetrained" },
      장어: { amountId: "eel-amount", retrainId: "eel-model-isRetrained" },
    };

    gallery.innerHTML = ""; // 갤러리 초기화
    selectedImageContainer.innerHTML = ""; // 선택 이미지 초기화

    for (const fishName of Object.keys(fishMap)) {
      // 예측 총량
      const totalPredictionKey = `${fishName}_1주_발주량 예측`;
      const retrainKey = `${fishName}_재학습_여부`;
      const imgKey = `${fishName}_결과_이미지`;

      // 예측값, 재학습 여부 테이블에 넣기
      const amountElem = document.getElementById(fishMap[fishName].amountId);
      const retrainElem = document.getElementById(fishMap[fishName].retrainId);

      if (amountElem) {
        amountElem.textContent = result[totalPredictionKey] ?? "-";
      }
      if (retrainElem) {
        retrainElem.textContent = result[retrainKey] === 1 ? "예" : "아니오";
      }

      // 이미지 갤러리에 추가
      if (result[imgKey]) {
        const fig = document.createElement("figure");
        fig.className = "effect-julia item";
        fig.innerHTML = `<img src="${result[imgKey]}" alt="${fishName} 예측 결과" style="cursor: pointer; max-width: 100%; height: auto;" />`;
        gallery.appendChild(fig);

        // 클릭 시 선택 이미지에 표시하고 섹션4로 스크롤
        const img = fig.querySelector("img");
        if (img) {
          img.addEventListener("click", () => {
            selectedImageContainer.innerHTML = `
              <img src="${img.src}" alt="${img.alt}" class="img-fluid mb-3" style="max-width: 100%; border-radius: 10px;" />
            `;
            document.getElementById("section-4").scrollIntoView({ behavior: "smooth" });
          });
        }
      }
    }
  }

  // 업로드 버튼 클릭 이벤트
  uploadButton.addEventListener("click", () => {
    if (!selectedFile) return;

    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://127.0.0.1:8001/upload", true);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        progressBar.style.width = percent + "%";
        progressBar.textContent = percent + "%";
        uploadPercentText.textContent = percent.toString();
        fileSizeText.textContent = `${formatBytes(e.loaded)} of ${formatBytes(e.total)}`;
      }
    });

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const result = JSON.parse(xhr.responseText);
          displayResults(result);
          uploadButton.textContent = "DONE";
          uploadButton.classList.remove("btn-primary");
          uploadButton.classList.add("btn-success");
        } catch (err) {
          alert("서버 응답 처리 중 오류가 발생했습니다.");
          console.error(err);
          uploadButton.disabled = false;
          uploadButton.textContent = "Upload";
        }
      } else {
        alert("업로드 실패: " + xhr.statusText);
        uploadButton.disabled = false;
        uploadButton.textContent = "Upload";
      }
    };

    xhr.onerror = () => {
      alert("업로드 중 오류가 발생했습니다.");
      progressBar.style.width = "0%";
      progressBar.textContent = "0%";
      fileSizeText.textContent = "0 MB";
      uploadButton.disabled = false;
      uploadButton.textContent = "Upload";
    };

    xhr.send(formData);
  });

  // 다운로드 버튼 클릭 시 선택 이미지 다운로드
  downloadImageButton.addEventListener("click", () => {
    const img = selectedImageContainer.querySelector("img");
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
