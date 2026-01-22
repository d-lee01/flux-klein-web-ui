document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('prompt');
    const modelSelect = document.getElementById('model-select');
    const stepsInput = document.getElementById('steps');
    const stepsVal = document.getElementById('steps-val');
    const guidanceInput = document.getElementById('guidance');
    const guidanceVal = document.getElementById('guidance-val');
    const generateBtn = document.getElementById('generate-btn');
    const resultImage = document.getElementById('result-image');
    const imagePlaceholder = document.getElementById('image-placeholder');
    const genTimeDisplay = document.getElementById('gen-time');
    const timeVal = document.getElementById('time-val');

    // Dimension buttons
    const dimBtns = document.querySelectorAll('.dim-btn');
    let currentWidth = 1024;
    let currentHeight = 1024;

    // Sliders
    stepsInput.addEventListener('input', (e) => stepsVal.textContent = e.target.value);
    guidanceInput.addEventListener('input', (e) => guidanceVal.textContent = e.target.value);

    // Dimensions
    dimBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            dimBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentWidth = parseInt(btn.dataset.w);
            currentHeight = parseInt(btn.dataset.h);
        });
    });

    // Generate
    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        // UI State: Loading
        generateBtn.classList.add('loading');
        generateBtn.disabled = true;
        resultImage.hidden = true;
        imagePlaceholder.style.display = 'flex';
        genTimeDisplay.hidden = true;

        const startTime = Date.now();

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    model_name: modelSelect.value,
                    width: currentWidth,
                    height: currentHeight,
                    num_steps: parseInt(stepsInput.value),
                    guidance: parseFloat(guidanceInput.value)
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Generation failed');
            }

            const data = await response.json();

            // Show result
            resultImage.src = `data:image/png;base64,${data.image_base64}`;
            resultImage.hidden = false;
            imagePlaceholder.style.display = 'none';

            // Time
            const duration = ((Date.now() - startTime) / 1000).toFixed(2);
            timeVal.textContent = duration;
            genTimeDisplay.hidden = false;

        } catch (error) {
            console.error(error);
            alert(`Error: ${error.message}`);
        } finally {
            generateBtn.classList.remove('loading');
            generateBtn.disabled = false;
        }
    });
});
