document.addEventListener('DOMContentLoaded', function() {
    // Handle form submission and show loading modal
    const form = document.getElementById('uploadForm');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    
    if (form) {
        form.addEventListener('submit', function(e) {
            // Basic client-side validation
            const fileInput = document.getElementById('fileUpload');
            const radioInputs = document.querySelectorAll('input[name="inputType"]');
            let inputTypeSelected = false;
            
            radioInputs.forEach(input => {
                if (input.checked) {
                    inputTypeSelected = true;
                }
            });
            
            if (!inputTypeSelected) {
                e.preventDefault();
                alert('Please select an input type (Image or Video)');
                return;
            }
            
            if (!fileInput.value) {
                e.preventDefault();
                alert('Please select a file to upload');
                return;
            }
            
            // Check file size (16MB max)
            const maxSize = 16 * 1024 * 1024; // 16MB in bytes
            if (fileInput.files[0] && fileInput.files[0].size > maxSize) {
                e.preventDefault();
                alert('File size exceeds the maximum allowed (16MB)');
                return;
            }
            
            // Check file format based on selected input type
            const fileName = fileInput.value.toLowerCase();
            const imageType = document.getElementById('imageInput').checked;
            const videoType = document.getElementById('videoInput').checked;
            
            if (imageType && !(/\.(jpg|jpeg|png)$/.test(fileName))) {
                e.preventDefault();
                alert('Please select a valid image file (jpg, jpeg, or png)');
                return;
            }
            
            if (videoType && !(/\.(mp4|avi|mov)$/.test(fileName))) {
                e.preventDefault();
                alert('Please select a valid video file (mp4, avi, or mov)');
                return;
            }
            
            // If validation passes, show loading modal
            loadingModal.show();
            
            // Disable the submit button to prevent multiple submissions
            document.getElementById('submitBtn').disabled = true;
        });
    }
    
    // Handle file input change to validate file type
    const fileInput = document.getElementById('fileUpload');
    const imageInput = document.getElementById('imageInput');
    const videoInput = document.getElementById('videoInput');
    
    if (fileInput && imageInput && videoInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.value.toLowerCase();
            
            if (this.files.length > 0) {
                // Auto-select input type based on file extension
                if (/\.(jpg|jpeg|png)$/.test(fileName)) {
                    imageInput.checked = true;
                } else if (/\.(mp4|avi|mov)$/.test(fileName)) {
                    videoInput.checked = true;
                }
            }
        });
    }
    
    // Dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});
