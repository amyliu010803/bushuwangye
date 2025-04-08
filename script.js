/**
 * @typedef {Object} OCRResult
 * @property {string} text - OCR识别的文本
 * @property {number} confidence - 识别置信度
 */

/**
 * @typedef {Object} ProcessedImage
 * @property {string} dataUrl - 处理后的图片数据URL
 * @property {number} width - 图片宽度
 * @property {number} height - 图片高度
 */

// 全局变量
let currentFiles = [];
let processedImages = [];
let ocrResults = [];
let isTrainingDataLoaded = false;
let currentPageIndex = 0;
let cropBoxes = [];
let currentLayoutType = 'vertical';
let isDrawingCrop = false;
let startX = 0;
let startY = 0;
let originalImageScale = 1;
let currentPreviewImage = null;
let tesseractWorker = null;
let cropSettings = {
    top: 0,
    bottom: 0,
    left: 0,
    right: 0
};

// 常量
const TRAINING_DATA_PATH = './tessdata';
const DEEPSEEK_API_KEY = 'sk-6bc91d9c3c75469ea2c01f8187a9c29d';
const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions';

// DOM元素
const fileInput = document.getElementById('fileInput');
const progressBar = document.getElementById('progressBar');
const progressStatus = document.getElementById('progressStatus');
const imagePreview = document.getElementById('imagePreview');
const ocrResult = document.getElementById('ocrResult');
const downloadTxtBtn = document.getElementById('downloadTxtBtn');
const downloadDocBtn = document.getElementById('downloadDocBtn');
const clearBtn = document.getElementById('clearBtn');
const cropSettingsBtn = document.getElementById('cropSettingsBtn');
const cropModal = document.getElementById('cropModal');
const applyCropBtn = document.getElementById('applyCropBtn');
const cancelCropBtn = document.getElementById('cancelCropBtn');
const applyToAllBtn = document.getElementById('applyToAllBtn');
const clearCurrentPageBtn = document.getElementById('clearCurrentPageBtn');

// 初始化PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

/**
 * 显示OCR结果到页面
 * @param {OCRResult} result - OCR结果对象
 */
function displayResult(result) {
    const resultElement = document.getElementById('ocrResult');
    // 设置为可编辑
    resultElement.setAttribute('contenteditable', 'true');
    resultElement.textContent = result.text;
    
    // 如果需要，添加基本的编辑功能
    if (result.isOptimized) {
        resultElement.classList.add('optimized-text');
    }
    
    // 设置原始文本
    resultElement.dataset.originalText = result.text;
}

/**
 * 处理OCR结果并显示
 * @param {OCRResult[]} results - OCR结果数组
 */
function processFileResult(results) {
    // 显示每个页面的结果
    const pageContainer = document.getElementById('pageContainer');
    pageContainer.innerHTML = '';
    
    // 处理所有页面的识别结果
    let combinedText = '';
    let totalConfidence = 0;
    let hasOptimizedResults = false;
    
    results.forEach((result, index) => {
        // 添加本页OCR结果到组合文本
        combinedText += result.text;
        totalConfidence += result.confidence;
        
        if (result.isOptimized) {
            hasOptimizedResults = true;
        }
        
        // 如果不是最后一页，添加换页符
        if (index < results.length - 1) {
            combinedText += '\n\n----- 新页面 -----\n\n';
        }
    });
    
    // 计算平均置信度
    const averageConfidence = totalConfidence / results.length;
    
    // 更新页面计数
    document.getElementById('pageCount').textContent = `已处理 ${results.length} 页`;
    
    // 启用下载按钮
    if (downloadTxtBtn) {
        downloadTxtBtn.disabled = false;
        downloadTxtBtn.style.opacity = 1;
    }
    if (downloadDocBtn) {
        downloadDocBtn.disabled = false;
        downloadDocBtn.style.opacity = 1;
    }
    
    // 记录当前的结果用于下载
    currentOCRText = combinedText;
}

/**
 * 图像预处理
 * @param {HTMLCanvasElement} canvas - 画布元素
 * @param {CanvasRenderingContext2D} ctx - 画布上下文
 */
function preprocessImage(canvas, ctx) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // 计算图像的平均亮度和标准差
    let totalBrightness = 0;
    const brightnesses = [];
    
    for (let i = 0; i < data.length; i += 4) {
        const brightness = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
        totalBrightness += brightness;
        brightnesses.push(brightness);
    }
    
    const avgBrightness = totalBrightness / brightnesses.length;
    
    // 计算标准差
    let totalVariance = 0;
    for (const brightness of brightnesses) {
        totalVariance += Math.pow(brightness - avgBrightness, 2);
    }
    const stdDev = Math.sqrt(totalVariance / brightnesses.length);
    
    // 自适应对比度增强
    const contrastFactor = 1.5; // 基础对比度增强系数
    const brightnessFactor = 15; // 基础亮度调整
    
    // 分区域处理图像
    const blockSize = Math.floor(canvas.width / 5); // 将图像分为5列
    
    for (let x = 0; x < canvas.width; x++) {
        // 确定当前列的位置
        const column = Math.floor(x / blockSize);
        // 左侧区域（可能包含大标题）使用更强的对比度
        const localContrastFactor = column === 0 ? contrastFactor * 1.3 : contrastFactor;
        
        for (let y = 0; y < canvas.height; y++) {
            const i = (y * canvas.width + x) * 4;
            const brightness = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
            
            // 自适应对比度调整
            const factor = (259 * (localContrastFactor + 255)) / (255 * (259 - localContrastFactor));
            let newValue = factor * (brightness - 128) + 128 + brightnessFactor;
            
            // 限制在有效范围内
            newValue = Math.max(0, Math.min(255, newValue));
            
            data[i] = newValue;
            data[i + 1] = newValue;
            data[i + 2] = newValue;
        }
    }
    
    // 应用自适应阈值进行二值化
    const threshold = calculateOtsuThreshold(data);
    for (let i = 0; i < data.length; i += 4) {
        // 左侧区域使用较低的阈值，以更好地保留大标题
        const x = (i/4) % canvas.width;
        const column = Math.floor(x / blockSize);
        const localThreshold = column === 0 ? threshold * 0.9 : threshold;
        
        const value = data[i] < localThreshold ? 0 : 255;
        data[i] = value;
        data[i + 1] = value;
        data[i + 2] = value;
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // 应用锐化
    const sharpKernel = [
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    ];
    
    // 创建临时画布来存储锐化后的图像
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // 复制原始图像到临时画布
    tempCtx.drawImage(canvas, 0, 0);
    
    // 应用锐化卷积核
    applyConvolution(canvas, ctx, tempCanvas, tempCtx, sharpKernel);
    
    // 最后应用额外的对比度增强
    ctx.filter = 'contrast(1.3) brightness(1.1)';
    ctx.drawImage(canvas, 0, 0);
}

/**
 * 应用卷积操作到图像
 * @param {HTMLCanvasElement} canvas - 目标画布
 * @param {CanvasRenderingContext2D} ctx - 目标画布上下文
 * @param {HTMLCanvasElement} tempCanvas - 临时画布
 * @param {CanvasRenderingContext2D} tempCtx - 临时画布上下文
 * @param {Array} kernel - 卷积核
 */
function applyConvolution(canvas, ctx, tempCanvas, tempCtx, kernel) {
    const tempImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const tempData = tempImageData.data;
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // 卷积操作
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const pixelIndex = (y * width + x) * 4;
            
            let r = 0, g = 0, b = 0;
            
            // 应用3x3卷积核
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const kernelIndex = (ky + 1) * 3 + (kx + 1);
                    const targetPixelIndex = ((y + ky) * width + (x + kx)) * 4;
                    
                    r += tempData[targetPixelIndex] * kernel[kernelIndex];
                    g += tempData[targetPixelIndex + 1] * kernel[kernelIndex];
                    b += tempData[targetPixelIndex + 2] * kernel[kernelIndex];
                }
            }
            
            // 限制在有效范围内
            data[pixelIndex] = Math.max(0, Math.min(255, r));
            data[pixelIndex + 1] = Math.max(0, Math.min(255, g));
            data[pixelIndex + 2] = Math.max(0, Math.min(255, b));
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
}

/**
 * 计算大津阈值
 * @param {Uint8ClampedArray} data - 图像数据
 * @returns {number} - 计算得到的阈值
 */
function calculateOtsuThreshold(data) {
    const histogram = new Array(256).fill(0);
    let pixelCount = 0;
    
    // 计算直方图
    for (let i = 0; i < data.length; i += 4) {
        histogram[data[i]]++;
        pixelCount++;
    }
    
    let sumTotal = 0;
    for (let i = 0; i < 256; i++) {
        sumTotal += i * histogram[i];
    }
    
    let sumBackground = 0;
    let weightBackground = 0;
    let weightForeground = 0;
    let maxVariance = 0;
    let threshold = 0;
    
    // 计算最佳阈值
    for (let t = 0; t < 256; t++) {
        weightBackground += histogram[t];
        if (weightBackground === 0) continue;
        
        weightForeground = pixelCount - weightBackground;
        if (weightForeground === 0) break;
        
        sumBackground += t * histogram[t];
        
        const meanBackground = sumBackground / weightBackground;
        const meanForeground = (sumTotal - sumBackground) / weightForeground;
        
        const variance = weightBackground * weightForeground * 
                        Math.pow(meanBackground - meanForeground, 2);
        
        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = t;
        }
    }
    
    return threshold;
}

/**
 * 初始化OCR Worker
 * @returns {Promise<void>}
 */
async function initializeOCRWorker() {
    try {
        if (tesseractWorker) {
            await tesseractWorker.terminate();
            tesseractWorker = null;
        }
        
        console.log('正在初始化Tesseract Worker...');
        updateProgress(0, '正在初始化OCR引擎...');
        
        // 创建worker实例
        tesseractWorker = await Tesseract.createWorker();
        updateProgress(30, '正在加载OCR引擎...');
        
        // 加载日语训练数据
        await tesseractWorker.loadLanguage('jpn+jpn_vert');
        updateProgress(60, '正在加载训练数据...');
        
        // 初始化
        await tesseractWorker.initialize('jpn+jpn_vert');
        updateProgress(80, '正在初始化OCR引擎...');
        
        // 设置识别参数 - 基础日文模式
        await tesseractWorker.setParameters({
            tessedit_pageseg_mode: '5',  // 使用垂直文本模式
            preserve_interword_spaces: '1',
            tessedit_char_blacklist: '',
            tessedit_enable_dict_correction: '1',
            segment_nonalphabetic_script: '1',
            tessedit_ocr_engine_mode: '2',  // 使用LSTM引擎
            lstm_choice_mode: '2',  // 更准确的LSTM模式
            textord_tabfind_vertical_text: '1',  // 启用垂直文本检测
            textord_vertical_text: '1',  // 强制垂直文本模式
            textord_rotation_type: '1'  // 强制垂直模式
        });
        
        console.log('Tesseract Worker初始化完成');
        updateProgress(100, '初始化完成，可以开始使用');
    } catch (error) {
        console.error('Tesseract Worker初始化失败:', error);
        updateProgress(0, 'OCR引擎初始化失败', error.message);
        throw new Error(`OCR初始化失败: ${error.message}`);
    }
}

/**
 * 判断文本是否更可能是日文而不是英文
 * @param {string} text - 要分析的文本
 * @returns {boolean} - 如果文本更可能是日文则返回true
 */
function isMoreLikelyJapanese(text) {
    if (!text) return false;
    
    // 日文字符的正则表达式 (包括平假名、片假名和汉字)
    const japaneseRegex = /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]/g;
    
    // 英文字符的正则表达式
    const englishRegex = /[a-zA-Z]/g;
    
    // 计算日文和英文字符的数量
    const japaneseMatches = text.match(japaneseRegex) || [];
    const englishMatches = text.match(englishRegex) || [];
    
    const japaneseCount = japaneseMatches.length;
    const englishCount = englishMatches.length;
    
    // 计算总字符数（不包括空格和标点）
    const totalCount = text.replace(/[\s\p{P}]/gu, '').length;
    
    // 如果总字符不足，不做判断
    if (totalCount < 5) return true;  // 默认假设是日文
    
    // 计算日文字符占比
    const japaneseRatio = japaneseCount / totalCount;
    
    // 如果日文字符占比超过40%，认为是日文
    return japaneseRatio > 0.4;
}

/**
 * 检测并裁剪页边距
 * @param {HTMLCanvasElement} canvas - 画布元素
 * @param {CanvasRenderingContext2D} ctx - 画布上下文
 * @returns {Object} 裁剪后的区域信息
 */
function detectAndCropMargins(canvas, ctx) {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // 计算图像的平均亮度和标准差
    let totalBrightness = 0;
    const brightnesses = [];
    
    for (let i = 0; i < data.length; i += 4) {
        const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
        totalBrightness += brightness;
        brightnesses.push(brightness);
    }
    
    const avgBrightness = totalBrightness / brightnesses.length;
    
    // 计算标准差
    let totalVariance = 0;
    for (const brightness of brightnesses) {
        totalVariance += Math.pow(brightness - avgBrightness, 2);
    }
    const stdDev = Math.sqrt(totalVariance / brightnesses.length);
    
    // 动态计算阈值：平均亮度 +/- 2个标准差
    const threshold = avgBrightness - (2 * stdDev);
    
    // 初始化边界值
    let left = canvas.width;
    let right = 0;
    let top = canvas.height;
    let bottom = 0;
    
    // 将图像分成网格进行分析
    const gridSize = 20; // 网格大小
    const gridRows = Math.ceil(canvas.height / gridSize);
    const gridCols = Math.ceil(canvas.width / gridSize);
    const grid = Array(gridRows).fill().map(() => Array(gridCols).fill(0));
    
    // 计算每个网格的文本密度
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const i = (y * canvas.width + x) * 4;
            const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
            
            if (brightness < threshold) {
                const gridRow = Math.floor(y / gridSize);
                const gridCol = Math.floor(x / gridSize);
                grid[gridRow][gridCol]++;
            }
        }
    }
    
    // 计算每个网格的平均密度
    const densities = grid.map(row => row.map(count => count / (gridSize * gridSize)));
    
    // 计算整体平均密度和标准差
    let totalDensity = 0;
    let densityCount = 0;
    for (const row of densities) {
        for (const density of row) {
            if (density > 0) {
                totalDensity += density;
                densityCount++;
            }
        }
    }
    const avgDensity = totalDensity / densityCount;
    
    // 计算密度标准差
    let densityVariance = 0;
    for (const row of densities) {
        for (const density of row) {
            if (density > 0) {
                densityVariance += Math.pow(density - avgDensity, 2);
            }
        }
    }
    const densityStdDev = Math.sqrt(densityVariance / densityCount);
    
    // 设置密度阈值：平均密度的30%
    const densityThreshold = avgDensity * 0.3;
    
    // 标记需要保留的区域
    const validGrid = Array(gridRows).fill().map(() => Array(gridCols).fill(false));
    
    // 使用滑动窗口检测连续的文本区域
    const windowSize = 3;
    for (let i = 0; i < gridRows - windowSize; i++) {
        for (let j = 0; j < gridCols - windowSize; j++) {
            let windowDensity = 0;
            let maxDensity = 0;
            
            // 计算窗口内的平均密度和最大密度
            for (let wi = 0; wi < windowSize; wi++) {
                for (let wj = 0; wj < windowSize; wj++) {
                    windowDensity += densities[i + wi][j + wj];
                    maxDensity = Math.max(maxDensity, densities[i + wi][j + wj]);
                }
            }
            windowDensity /= (windowSize * windowSize);
            
            // 如果窗口内的密度分布合理，标记为有效区域
            if (windowDensity > densityThreshold && maxDensity < avgDensity + densityStdDev) {
                for (let wi = 0; wi < windowSize; wi++) {
                    for (let wj = 0; wj < windowSize; wj++) {
                        validGrid[i + wi][j + wj] = true;
                    }
                }
            }
        }
    }
    
    // 基于有效网格确定边界
    let validTop = -1, validBottom = -1;
    let validLeft = gridCols, validRight = 0;
    
    // 查找垂直边界
    for (let i = 0; i < gridRows; i++) {
        let hasValidCell = false;
        for (let j = 0; j < gridCols; j++) {
            if (validGrid[i][j]) {
                hasValidCell = true;
                validLeft = Math.min(validLeft, j);
                validRight = Math.max(validRight, j);
            }
        }
        if (hasValidCell) {
            if (validTop === -1) validTop = i;
            validBottom = i;
        }
    }
    
    // 转换回像素坐标
    if (validTop !== -1 && validLeft !== gridCols) {
        top = validTop * gridSize;
        bottom = (validBottom + 1) * gridSize;
        left = validLeft * gridSize;
        right = (validRight + 1) * gridSize;
        
        // 添加适当的边距
        const paddingX = Math.min(50, Math.max(20, (right - left) * 0.05));
        const paddingY = Math.min(50, Math.max(20, (bottom - top) * 0.05));
        
        left = Math.max(0, left - paddingX);
        right = Math.min(canvas.width, right + paddingX);
        top = Math.max(0, top - paddingY);
        bottom = Math.min(canvas.height, bottom + paddingY);
        
        // 执行裁剪
        const width = right - left;
        const height = bottom - top;
        
        if (width > canvas.width * 0.1 && height > canvas.height * 0.1) {
            const newCanvas = document.createElement('canvas');
            newCanvas.width = width;
            newCanvas.height = height;
            const newCtx = newCanvas.getContext('2d');
            
            newCtx.imageSmoothingEnabled = true;
            newCtx.imageSmoothingQuality = 'high';
            
            newCtx.drawImage(canvas, 
                left, top, width, height,
                0, 0, width, height
            );
            
            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(newCanvas, 0, 0);
            
            console.log('智能边距裁剪完成:', {
                原始大小: { width: canvas.width, height: canvas.height },
                裁剪区域: { left, top, right, bottom },
                裁剪后大小: { width, height },
                平均密度: avgDensity,
                密度阈值: densityThreshold
            });
        }
    } else {
        console.log('未检测到有效的文本区域，保持原图不变');
    }

    return {
        left, right, top, bottom,
        width: right - left,
        height: bottom - top
    };
}

/**
 * 处理图片文件
 * @param {File} file - 图片文件
 * @returns {Promise<ProcessedImage>}
 */
async function processImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const img = new Image();
                img.onload = async () => {
                    try {
                        const canvas = document.createElement('canvas');
                        const ctx = canvas.getContext('2d');
                        
                        // 设置画布大小
                        canvas.width = img.width;
                        canvas.height = img.height;
                        
                        // 直接绘制图片
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                        
                        resolve({
                            dataUrl: canvas.toDataURL('image/png'),
                            width: canvas.width,
                            height: canvas.height
                        });
                    } catch (error) {
                        reject(new Error(`图片处理失败: ${error.message}`));
                    }
                };
                img.onerror = () => reject(new Error('图片加载失败'));
                img.src = e.target.result;
            } catch (error) {
                reject(new Error(`图片处理过程出错: ${error.message}`));
            }
        };
        reader.onerror = () => reject(new Error('文件读取失败'));
        reader.readAsDataURL(file);
    });
}

/**
 * 处理PDF文件
 * @param {File} file - PDF文件
 * @returns {Promise<ProcessedImage[]>}
 */
async function processPDF(file) {
    try {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({
            data: arrayBuffer,
            cMapUrl: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/cmaps/',
            cMapPacked: true,
            standardFontDataUrl: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/standard_fonts/'
        }).promise;
        
        console.log(`PDF加载成功，共 ${pdf.numPages} 页`);
        const images = [];
        
        for (let i = 1; i <= pdf.numPages; i++) {
            try {
                console.log(`开始处理第 ${i} 页...`);
                const page = await pdf.getPage(i);
                
                // 获取页面的原始尺寸
                const viewport = page.getViewport({ scale: 1.0 });
                
                // 计算合适的缩放比例，确保图像质量足够好
                const scale = Math.max(2.0, 2400 / Math.max(viewport.width, viewport.height));
                const scaledViewport = page.getViewport({ scale: scale });
                
                // 创建canvas
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = scaledViewport.height;
                canvas.width = scaledViewport.width;
                
                console.log(`渲染第 ${i} 页，尺寸: ${canvas.width}x${canvas.height}`);
                
                // 渲染PDF页面
                const renderContext = {
                    canvasContext: context,
                    viewport: scaledViewport,
                    enableWebGL: true,
                    renderInteractiveForms: false
                };
                
                await page.render(renderContext).promise;
                
                // 将canvas转换为图像数据
                const imageData = canvas.toDataURL('image/png');
                images.push({
                    dataUrl: imageData,
                    width: canvas.width,
                    height: canvas.height
                });
                
                updateProgress((i / pdf.numPages) * 100, `正在处理PDF第 ${i} 页，共 ${pdf.numPages} 页...`);
                console.log(`第 ${i} 页处理完成`);
            } catch (error) {
                console.error(`处理PDF第 ${i} 页时出错:`, error);
                throw new Error(`PDF第 ${i} 页处理失败: ${error.message}`);
            }
        }
        
        console.log('PDF处理完成，共处理', images.length, '页');
        return images;
    } catch (error) {
        console.error('PDF处理失败:', error);
        throw new Error(`PDF处理失败: ${error.message}`);
    }
}

/**
 * 智能识别日语文本中的段落结构
 * @param {string} text - OCR识别的原始文本
 * @returns {string} - 格式化后带有段落结构的文本
 */
function detectParagraphs(text) {
    if (!text || typeof text !== 'string') {
        return text;
    }
    
    // 将垂直文本转换为水平文本格式
    // 移除多余的空格和换行，但保留段落间的空行
    let horizontalText = text.replace(/\s{2,}/g, '\n\n').replace(/\s+/g, ' ').trim();
    
    // 日语句子结束标记
    const sentenceEnders = ['。', '！', '？', '…', '.', '!', '?'];
    const dialogueMarkers = ['「', '」', '『', '』', '"', '"'];
    const paragraphStarters = ['　', '    ', '  ']; // 全角空格和缩进
    
    // 分段策略：
    // 1. 在句子结束符+空格后添加段落分隔
    // 2. 在特定的对话标记或段落起始符号处添加段落分隔
    // 3. 保持引号内的对话为一个整体
    
    let formattedText = '';
    let inQuote = false; // 标记是否在引号内
    let sentenceBuffer = '';
    let paragraphs = [];
    
    // 第一步：按基本规则分割成初步段落
    for (let i = 0; i < horizontalText.length; i++) {
        const char = horizontalText[i];
        
        // 处理引号内文本
        if (dialogueMarkers.includes(char)) {
            if (char === '「' || char === '『' || char === '"') {
                inQuote = true;
            } else if (char === '」' || char === '』' || char === '"') {
                inQuote = false;
                // 引号结束后如果有句子结束符，这里可能是段落结束
                if (i + 1 < horizontalText.length && sentenceEnders.includes(horizontalText[i+1])) {
                    sentenceBuffer += char;
                    i++; // 跳过下一个句子结束符
                    sentenceBuffer += horizontalText[i];
                    paragraphs.push(sentenceBuffer);
                    sentenceBuffer = '';
                    continue;
                }
            }
        }
        
        sentenceBuffer += char;
        
        // 在非引号内，检查是否是句子结束标记
        if (!inQuote && sentenceEnders.includes(char)) {
            // 句子结束，检查是否应该结束段落
            // 如果后面跟着空格或者已经是文本结尾，这可能是段落结束
            if (i + 1 >= horizontalText.length || 
                horizontalText[i+1] === ' ' || 
                paragraphStarters.some(starter => horizontalText.substring(i+1).startsWith(starter))) {
                paragraphs.push(sentenceBuffer);
                sentenceBuffer = '';
            }
        }
    }
    
    // 添加最后一个句子（如果有）
    if (sentenceBuffer.length > 0) {
        paragraphs.push(sentenceBuffer);
    }
    
    // 第二步：合并相关的句子形成更合理的段落
    let finalParagraphs = [];
    let currentParagraph = '';
    
    for (let i = 0; i < paragraphs.length; i++) {
        const paragraph = paragraphs[i].trim();
        
        // 如果当前段落为空，直接使用这个段落
        if (currentParagraph === '') {
            currentParagraph = paragraph;
            continue;
        }
        
        // 检查这个段落是否应该合并到当前段落
        // 1. 如果它很短（比如不到10个字符）
        // 2. 如果它以特定的连接词开始
        // 3. 如果当前段落以连接词或省略号结束
        const shortParagraph = paragraph.length < 10;
        const startsWithConnective = /^[それでもしかしただし]/.test(paragraph);
        const currentEndsWithConnective = /[しかしまた]$/.test(currentParagraph);
        
        if (shortParagraph || startsWithConnective || currentEndsWithConnective) {
            currentParagraph += paragraph;
        } else {
            finalParagraphs.push(currentParagraph);
            currentParagraph = paragraph;
        }
    }
    
    // 添加最后一个段落
    if (currentParagraph.length > 0) {
        finalParagraphs.push(currentParagraph);
    }
    
    // 最终处理：将段落用换行符连接
    formattedText = finalParagraphs.join('\n\n');
    
    // 特殊处理：确保对话格式正确
    formattedText = formattedText.replace(/([「『])(.*?)([」』])/g, (match, open, content, close) => {
        // 保持对话内容在同一段落
        return open + content.replace(/\n\n/g, ' ') + close;
    });
    
    return formattedText;
}

/**
 * 执行OCR识别
 * @param {ProcessedImage} image - 处理后的图片
 * @returns {Promise<OCRResult>}
 */
async function performOCR(image) {
    try {
        if (!tesseractWorker) {
            await initializeOCRWorker();
        }

        console.log('开始OCR识别...');
        updateProgress(0, '执行文字识别...');
        
        // 设置基础识别参数
        await tesseractWorker.setParameters({
            tessedit_pageseg_mode: '5',  // 使用垂直文本模式
            preserve_interword_spaces: '1',
            tessedit_char_blacklist: '',
            tessedit_enable_dict_correction: '1',
            segment_nonalphabetic_script: '1',
            tessedit_ocr_engine_mode: '2',  // 使用LSTM引擎
            lstm_choice_mode: '2',  // 更准确的LSTM模式
            textord_tabfind_vertical_text: '1',  // 启用垂直文本检测
            textord_vertical_text: '1',  // 强制垂直文本模式
            textord_rotation_type: '1'  // 强制垂直模式
        });
        
        // 执行OCR
        updateProgress(30, '文字识别中...');
        const result = await tesseractWorker.recognize(image.dataUrl);
        
        // 检查文本是否为空
        if (!result.data.text || result.data.text.trim() === '') {
            console.log('OCR未识别出文本');
            return {
                text: '',
                confidence: 0,
                isOptimized: false
            };
        }
        
        // 返回OCR结果
            return {
            text: result.data.text,
                confidence: result.data.confidence,
                isOptimized: false
            };
        
    } catch (error) {
        console.error('OCR识别失败:', error);
        updateProgress(0, 'OCR识别失败', error.message);
        throw error;
    }
}

/**
 * 使用DeepSeek API优化OCR结果
 * @param {string} text - 原始OCR文本
 * @returns {Promise<string>} - 优化后的文本
 */
async function optimizeWithDeepSeek(text) {
    try {
        // 如果文本为空或只包含空白字符，直接返回换行符
        if (!text || text.trim() === '') {
            console.log('输入文本为空，跳过优化');
            return '\n';
        }

        console.log('调用DeepSeek API优化文本...');
        console.log('原始文本长度:', text.length);
        
        // 将文本拆分为较小的块，确保不超过API限制
        const chunks = [];
        const maxChunkSize = 2000; // 字符数
        let currentIndex = 0;
        
        while (currentIndex < text.length) {
            const chunk = text.slice(currentIndex, currentIndex + maxChunkSize);
            chunks.push(chunk);
            currentIndex += maxChunkSize;
        }
        
        console.log(`文本已拆分为 ${chunks.length} 个块进行处理`);
        
        // 分别处理每个块
        const processedChunks = [];
        for (let i = 0; i < chunks.length; i++) {
            console.log(`处理第 ${i+1}/${chunks.length} 个文本块`);
            const chunk = chunks[i];
            
        const response = await fetch(DEEPSEEK_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${DEEPSEEK_API_KEY}`
            },
            body: JSON.stringify({
                model: "deepseek-chat",
                messages: [
                    {
                        "role": "system",
                            "content": `你是一个专业的日语竖排OCR文本校对专家。请校正文本中的错误并调整段落格式。

校对任务：
1. 修正OCR错误（如"大"被错识别为"太"、"は"被识别为"ば"等）
2. 删除多余的空格和换行，建立合理的段落
3. 修正标点符号错误（如书名号、双引号不对应等）

重要规则：
1. 不要增加或删除实质内容
2. 不要将假名转换为汉字
3. 不要改变文本的句子结构
4. 不要丢弃任何文本内容，即使是不完整的句子
5. 文本末尾的不完整内容必须原样保留

请直接返回校正后的完整文本，不添加任何注释或说明。确保输出包含全部内容，特别是末尾部分。`
                    },
                    {
                        "role": "user",
                            "content": `请校对以下OCR文本，调整段落，修正错误，但不要删除任何实质内容，特别是末尾内容必须完整保留：\n\n${chunk}`
                        }
                    ],
                    temperature: 0.3, // 适当提高温度以增强文本优化能力
                    max_tokens: 4000,
                    top_p: 0.2,       // 适当的top_p
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0
            })
        });

        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }

        const data = await response.json();
            console.log(`块 ${i+1} API响应:`, data);
        
        if (!data.choices || !data.choices[0] || !data.choices[0].message || !data.choices[0].message.content) {
                console.error('API返回的数据格式不正确:', data);
                throw new Error('API返回的数据格式不正确');
            }

            const processedChunk = data.choices[0].message.content.trim();
            processedChunks.push(processedChunk);
            console.log(`块 ${i+1} 处理完成，长度: ${processedChunk.length}`);
        }
        
        // 合并所有处理后的块
        const optimizedText = processedChunks.join('');
        console.log('所有块处理完成');
        console.log('原始文本长度:', text.length, '优化后文本长度:', optimizedText.length);
        
        // 检查是否丢失文本内容
        if (optimizedText.length < text.length * 0.8) {
            console.warn('警告：优化后的文本长度明显小于原始文本，可能丢失内容');
            console.log('返回原始文本以避免内容丢失');
            return text; // 当检测到可能丢失大量内容时，返回原始文本
        }
        
        return optimizedText;
    } catch (error) {
        console.error('DeepSeek API调用失败:', error);
        // 在API调用失败时，返回原始文本以避免丢失内容
        console.log('返回原始未处理文本');
        return text;  
    }
}

/**
 * 生成文本文件
 * @param {Array} results - OCR结果数组
 * @returns {Blob} 包含处理后文本的Blob对象
 */
function generateTextFile(results) {
    let content = '';
    
    results.forEach((result, index) => {
        if (result && result.text && result.text.trim()) {
            // 直接添加 AI 优化后的文本
            content += result.text;
            
            // 在每页之间添加换行
            if (index < results.length - 1) {
                content += '\n\n';
            }
        }
    });
    
    return new Blob([content], { type: 'text/plain;charset=utf-8' });
}

/**
 * 显示错误信息
 * @param {string} message - 错误信息
 */
function showError(message) {
    console.error(message);
    updateProgress(0, message);
}

/**
 * 生成Word文档
 */
function generateWordFile() {
    try {
        // 检查是否有OCR结果
        if (!ocrResults || ocrResults.length === 0) {
            updateProgress(0, '没有可用的OCR结果');
            return;
        }

        // 检查 docx 是否可用
        if (typeof docx === 'undefined') {
            console.error('docx.js未正确加载');
            updateProgress(0, 'Word文档生成失败：docx.js未加载');
            return;
        }

        console.log('开始生成Word文档...');
        
        // 创建文档
        const doc = new docx.Document({
            sections: [{
                properties: {},
                children: ocrResults.flatMap((result, index) => {
                    const children = [];
                    
                    if (result.text && result.text.trim()) {
                        // 分段处理
                        const paragraphs = result.text.split(/\n\s*\n/);
                        
                        paragraphs.forEach(para => {
                            if (para.trim()) {
                                // 添加文本段落
                                children.push(new docx.Paragraph({
                                    children: [
                                        new docx.TextRun({
                                            text: para.trim(),
                                            color: "000000"
                                        })
                                    ]
                                }));
                            }
                        });
                        
                        // 如果不是最后一页，添加页码标记
                        if (index < ocrResults.length - 1) {
                            children.push(new docx.Paragraph({
                                children: [
                                    new docx.TextRun({
                                        text: `----------第 ${index + 2} 页----------`,
                                        color: "666666",
                                        size: 20
                                    })
                                ],
                                alignment: docx.AlignmentType.CENTER,
                                spacing: {
                                    before: 240,  // 添加上边距
                                    after: 240    // 添加下边距
                                }
                            }));
                        }
                    }
                    
                    return children;
                })
            }]
        });

        console.log('文档对象创建成功，准备生成blob...');

        // 生成并下载文档
        docx.Packer.toBlob(doc).then(blob => {
            console.log('Blob生成成功，准备下载...');
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ocr_result.docx';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            updateProgress(100, '文档生成完成');
        }).catch(error => {
            console.error('生成Word文档blob时出错:', error);
            updateProgress(0, '生成Word文档失败: ' + error.message);
        });
    } catch (error) {
        console.error('生成Word文档时出错:', error);
        updateProgress(0, '生成Word文档失败: ' + error.message);
    }
}

/**
 * 显示图片预览和OCR结果
 * @param {File[]} files - 原始文件数组
 * @param {OCRResult[]} results - OCR结果数组
 */
function displayPreviewAndResults(files, results) {
    const imagePreview = document.getElementById('imagePreview');
    const pageCount = document.getElementById('pageCount');
    
    if (!imagePreview) {
        console.error('未找到图片预览容器');
        return;
    }
    
    // 清空预览区域
    imagePreview.innerHTML = '';
    
    // 显示图片预览和结果
    processedImages.forEach((processedImage, index) => {
        const pageDiv = document.createElement('div');
        pageDiv.className = 'page-container';
        pageDiv.setAttribute('data-page', index + 1);
        
        // 图片预览部分（左侧）
        const previewContainer = document.createElement('div');
        previewContainer.className = 'preview-container';
        
        const pageNumber = document.createElement('p');
        pageNumber.className = 'page-number';
        pageNumber.textContent = `第 ${index + 1} 页`;
        
        const previewImg = document.createElement('img');
        previewImg.className = 'preview-image';
        previewImg.src = processedImage.dataUrl;
        previewImg.alt = `页面 ${index + 1}`;
        
        previewContainer.appendChild(pageNumber);
        previewContainer.appendChild(previewImg);
        
        // 结果容器部分（右侧）
        const resultContainer = document.createElement('div');
        resultContainer.className = 'result-container';
        
        // 如果有OCR结果，显示文本
        if (results && results[index]) {
            const result = results[index];
            
            // 创建结果工具栏（可以切换显示优化前后的结果）
            if (result.text && result.text.trim()) {
                const resultToolbar = document.createElement('div');
                resultToolbar.className = 'result-toolbar';
                resultToolbar.style.display = 'flex';
                resultToolbar.style.justifyContent = 'flex-end';
                resultToolbar.style.margin = '5px 0';
                resultToolbar.style.padding = '5px';
                resultToolbar.style.backgroundColor = '#f8f9fa';
                resultToolbar.style.borderRadius = '4px';
                
                // 如果存在原始OCR结果，添加切换按钮
                if (result.originalText) {
                    const toggleBtn = document.createElement('button');
                    toggleBtn.textContent = '查看原始OCR结果';
                    toggleBtn.className = 'toggle-btn';
                    toggleBtn.style.padding = '3px 8px';
                    toggleBtn.style.fontSize = '12px';
                    toggleBtn.style.backgroundColor = '#2196F3';
                    toggleBtn.style.color = 'white';
                    toggleBtn.style.border = 'none';
                    toggleBtn.style.borderRadius = '3px';
                    toggleBtn.style.cursor = 'pointer';
                    toggleBtn.dataset.showOriginal = 'false';
                    
                    // 添加切换逻辑
                    toggleBtn.addEventListener('click', function() {
                        const textElement = this.parentNode.nextElementSibling;
                        const isShowingOriginal = this.dataset.showOriginal === 'true';
                        
                        if (isShowingOriginal) {
                            // 切换到优化后的结果
                            textElement.textContent = result.text;
                            this.textContent = '查看原始OCR结果';
                            this.dataset.showOriginal = 'false';
                            this.style.backgroundColor = '#2196F3';
                        } else {
                            // 切换到原始OCR结果
                            textElement.textContent = result.originalText;
                            this.textContent = '查看优化后结果';
                            this.dataset.showOriginal = 'true';
                            this.style.backgroundColor = '#FF9800';
                        }
                    });
                    
                    resultToolbar.appendChild(toggleBtn);
                }
                
                resultContainer.appendChild(resultToolbar);
                
                // 添加文本容器
        const resultText = document.createElement('div');
        resultText.className = 'vertical-text';
        resultText.setAttribute('contenteditable', 'true');
                resultText.textContent = result.text;
                
                resultContainer.appendChild(resultText);
            } else {
                const resultText = document.createElement('div');
                resultText.className = 'vertical-text';
                resultText.textContent = '无识别结果';
                resultContainer.appendChild(resultText);
            }
        } else {
            const resultText = document.createElement('div');
            resultText.className = 'vertical-text';
            resultText.textContent = '等待识别...';
            resultContainer.appendChild(resultText);
        }
        
        // 将预览和结果添加到页面容器
        pageDiv.appendChild(previewContainer);
        pageDiv.appendChild(resultContainer);
        
        // 添加到图片预览区域
        imagePreview.appendChild(pageDiv);
    });
    
    // 更新页面计数
    if (pageCount) {
        if (processedImages.length > 0) {
            pageCount.textContent = `共 ${processedImages.length} 页`;
        } else {
            pageCount.textContent = '';
        }
    }
}

/**
 * 应用裁剪设置到图像
 * @param {HTMLCanvasElement} canvas - 画布元素
 * @param {CanvasRenderingContext2D} ctx - 画布上下文
 * @param {Object} settings - 裁剪设置
 * @returns {void}
 */
function applyCropToImage(canvas, ctx) {
    const width = canvas.width - (cropSettings.left + cropSettings.right);
    const height = canvas.height - (cropSettings.top + cropSettings.bottom);
    
    if (width <= 0 || height <= 0) {
        console.warn('裁剪区域无效，保持原图不变');
        return;
    }
    
    const imageData = ctx.getImageData(
        cropSettings.left,
        cropSettings.top,
        width,
        height
    );
    
    // 创建新画布并绘制裁剪后的图像
    const newCanvas = document.createElement('canvas');
    newCanvas.width = width;
    newCanvas.height = height;
    const newCtx = newCanvas.getContext('2d');
    
    newCtx.putImageData(imageData, 0, 0);
    
    // 将裁剪后的图像复制回原画布
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(newCanvas, 0, 0);
}

/**
 * 获取鼠标在画布上的精确坐标
 * @param {HTMLCanvasElement} canvas - 画布元素
 * @param {MouseEvent} event - 鼠标事件
 * @returns {Object} 包含x和y坐标的对象
 */
function getMousePos(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY
    };
}

// 初始化裁剪Canvas的事件监听器
function initCropCanvasEvents() {
    const previewCanvas = document.getElementById('cropPreview');
    const ctx = previewCanvas.getContext('2d');
    
    // 鼠标按下事件
    previewCanvas.addEventListener('mousedown', (e) => {
        if (!currentPreviewImage) return;
        
        const pos = getMousePos(previewCanvas, e);
        startX = pos.x;
        startY = pos.y;
        isDrawingCrop = true;
    });
    
    // 鼠标移动事件
    previewCanvas.addEventListener('mousemove', (e) => {
        if (!isDrawingCrop || !currentPreviewImage) return;
        
        const pos = getMousePos(previewCanvas, e);
        const currentX = pos.x;
        const currentY = pos.y;
        
        // 重新绘制图像和所有已有的框
        const img = new Image();
        img.onload = () => {
            ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
            ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
            
            // 绘制已有的裁剪框
            cropBoxes.forEach((box, index) => {
                drawCropBox(ctx, box.x, box.y, box.width, box.height, index + 1);
            });
            
            // 绘制当前正在绘制的框
            const width = currentX - startX;
            const height = currentY - startY;
            drawCropBox(ctx, startX, startY, width, height, cropBoxes.length + 1);
        };
        img.src = currentPreviewImage.dataUrl;
    });
    
    // 鼠标松开事件
    previewCanvas.addEventListener('mouseup', (e) => {
        if (!isDrawingCrop || !currentPreviewImage) return;
        
        const pos = getMousePos(previewCanvas, e);
        const endX = pos.x;
        const endY = pos.y;
        
        // 确保宽度和高度为正值
        const x = Math.min(startX, endX);
        const y = Math.min(startY, endY);
        const width = Math.abs(endX - startX);
        const height = Math.abs(endY - startY);
        
        // 如果框选区域太小，则忽略
        if (width < 10 || height < 10) {
            isDrawingCrop = false;
            return;
        }
        
        // 添加新的裁剪框（不需要缩放转换，因为已经在正确的坐标系中）
        cropBoxes.push({
            x: x,
            y: y,
            width: width,
            height: height
        });
        
        // 重新显示预览
        showCropPreview(currentPreviewImage);
        
        isDrawingCrop = false;
    });
    
    // 鼠标离开事件
    previewCanvas.addEventListener('mouseleave', () => {
        if (isDrawingCrop) {
            isDrawingCrop = false;
            showCropPreview(currentPreviewImage);
        }
    });
}

/**
 * 显示裁剪预览
 * @param {ProcessedImage} image - 处理后的图片
 */
function showCropPreview(image) {
    const previewCanvas = document.getElementById('cropPreview');
    const ctx = previewCanvas.getContext('2d');
    const img = new Image();
    
    currentPreviewImage = image; // 保存当前图像引用
    
    img.onload = () => {
        // 设置预览画布大小
        const maxSize = 800;
        previewCanvas.width = img.width;
        previewCanvas.height = img.height;
        
        // 清除画布
        ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
        
        // 绘制图像
        ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
        
        // 绘制已有的裁剪框
        if (cropBoxes && cropBoxes.length > 0) {
            cropBoxes.forEach((box, index) => {
                drawCropBox(ctx, box.x, box.y, box.width, box.height, index + 1);
            });
        }
    };
    
    img.src = image.dataUrl;
}

/**
 * 绘制裁剪框
 * @param {CanvasRenderingContext2D} ctx - 画布上下文
 * @param {number} x - 起始X坐标
 * @param {number} y - 起始Y坐标
 * @param {number} width - 宽度
 * @param {number} height - 高度
 * @param {number} index - 框选序号
 */
function drawCropBox(ctx, x, y, width, height, index) {
    // 生成不同的颜色
    const hue = (index * 137.5) % 360;
    ctx.strokeStyle = `hsl(${hue}, 70%, 50%)`;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    
    // 添加序号
    ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(index.toString(), x + width/2, y + height/2);
}

/**
 * 处理裁剪设置变化
 */
function handleCropSettingChange() {
    const topMarginInput = document.getElementById('topMargin');
    const bottomMarginInput = document.getElementById('bottomMargin');
    const leftMarginInput = document.getElementById('leftMargin');
    const rightMarginInput = document.getElementById('rightMargin');
    
    if (topMarginInput && bottomMarginInput && leftMarginInput && rightMarginInput) {
    cropSettings = {
        top: parseInt(topMarginInput.value) || 0,
        bottom: parseInt(bottomMarginInput.value) || 0,
        left: parseInt(leftMarginInput.value) || 0,
        right: parseInt(rightMarginInput.value) || 0
    };
    
        if (currentPreviewImage) {
            showCropPreview(currentPreviewImage);
        }
    }
}

// 添加裁剪设置相关的事件监听器
document.addEventListener('DOMContentLoaded', () => {
    const topMarginInput = document.getElementById('topMargin');
    const bottomMarginInput = document.getElementById('bottomMargin');
    const leftMarginInput = document.getElementById('leftMargin');
    const rightMarginInput = document.getElementById('rightMargin');
    
    if (topMarginInput && bottomMarginInput && leftMarginInput && rightMarginInput) {
[topMarginInput, bottomMarginInput, leftMarginInput, rightMarginInput].forEach(input => {
    input.addEventListener('input', handleCropSettingChange);
        });
    }
});

/**
 * 初始化PDF导航
 * @param {number} totalPages - PDF总页数
 */
function initPDFNavigation(totalPages) {
    const prevPageBtn = document.getElementById('prevPageBtn');
    const nextPageBtn = document.getElementById('nextPageBtn');
    const pageInfo = document.getElementById('pageInfo');
    
    // 更新页面信息显示
    function updatePageInfo() {
        if (pageInfo) {
            pageInfo.textContent = `第 ${currentPageIndex + 1} 页 / 共 ${totalPages} 页`;
        }
    }
    
    // 切换页面前保存当前页面的裁剪框设置
    function saveCurrentPageSettings() {
        if (processedImages[currentPageIndex]) {
            processedImages[currentPageIndex].cropSettings = {
                cropBoxes: cropBoxes.map(box => ({ ...box })),
                layoutType: currentLayoutType
            };
        }
    }
    
    // 加载目标页面的裁剪框设置
    function loadPageSettings(pageIndex) {
        if (processedImages[pageIndex]?.cropSettings) {
            cropBoxes = processedImages[pageIndex].cropSettings.cropBoxes.map(box => ({ ...box }));
            currentLayoutType = processedImages[pageIndex].cropSettings.layoutType;
    } else {
            cropBoxes = [];
        }
    }
    
    // 上一页按钮事件
    if (prevPageBtn) {
        prevPageBtn.addEventListener('click', () => {
            if (currentPageIndex > 0) {
                saveCurrentPageSettings();
                currentPageIndex--;
                loadPageSettings(currentPageIndex);
                
                if (processedImages[currentPageIndex]) {
                    showCropPreview(processedImages[currentPageIndex]);
                }
                updatePageInfo();
            }
        });
    }
    
    // 下一页按钮事件
    if (nextPageBtn) {
        nextPageBtn.addEventListener('click', () => {
            if (currentPageIndex < totalPages - 1) {
                saveCurrentPageSettings();
                currentPageIndex++;
                loadPageSettings(currentPageIndex);
                
                if (processedImages[currentPageIndex]) {
                    showCropPreview(processedImages[currentPageIndex]);
                }
                updatePageInfo();
            }
        });
    }
    
    // 初始化页面信息
    updatePageInfo();
}

// 初始化按钮事件
function initializeButtons() {
    // 清除按钮事件
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            // 清空所有数据
            currentFiles = [];
            processedImages = [];
            ocrResults = [];
            currentPageIndex = 0;
            cropBoxes = [];
            
            // 清空预览区域
            const imagePreview = document.getElementById('imagePreview');
            if (imagePreview) {
                imagePreview.innerHTML = '';
            }
            
            // 清空结果区域
            const ocrResult = document.getElementById('ocrResult');
            if (ocrResult) {
                ocrResult.textContent = '';
            }
            
            // 重置页面计数
            const pageCount = document.getElementById('pageCount');
            if (pageCount) {
                pageCount.textContent = '';
            }
            
            // 禁用下载按钮
            if (downloadTxtBtn) downloadTxtBtn.disabled = true;
            if (downloadDocBtn) downloadDocBtn.disabled = true;
            const downloadOriginalDocBtn = document.getElementById('downloadOriginalDocBtn');
            if (downloadOriginalDocBtn) downloadOriginalDocBtn.disabled = true;
            
            // 重置文件输入
            if (fileInput) {
                fileInput.value = '';
            }
            
            // 隐藏裁剪模态框
            if (cropModal) {
                cropModal.style.display = 'none';
            }
            
            // 清空进度条和状态
            updateProgress(0, '');
        });
    }
    
    // 清除当前页面的框选设置
    if (clearCurrentPageBtn) {
        clearCurrentPageBtn.addEventListener('click', () => {
            // 清空当前页面的裁剪框
            if (processedImages[currentPageIndex]) {
                cropBoxes = [];
                processedImages[currentPageIndex].cropSettings = {
                    cropBoxes: [],
                    layoutType: currentLayoutType
                };
                
                // 重新显示预览
                if (currentPreviewImage) {
                    showCropPreview(currentPreviewImage);
                }
            }
        });
    }
    
    // 将当前页设置应用到后续页
    if (applyToAllBtn) {
        applyToAllBtn.addEventListener('click', () => {
            const currentSettings = {
                cropBoxes: cropBoxes.map(box => ({ ...box })),
                layoutType: currentLayoutType
            };
            
            // 应用到后续页
            for (let i = currentPageIndex + 1; i < processedImages.length; i++) {
                processedImages[i].cropSettings = {
                    cropBoxes: currentSettings.cropBoxes.map(box => ({ ...box })),
                    layoutType: currentSettings.layoutType
                };
            }
            
            alert('设置已应用到后续页面');
        });
    }

    // 应用裁剪按钮事件
    if (applyCropBtn) {
applyCropBtn.addEventListener('click', async () => {
            if (cropModal) {
        cropModal.style.display = 'none';
            }
            
            // 保存当前页面的裁剪框设置
            if (processedImages[currentPageIndex]) {
                processedImages[currentPageIndex].cropSettings = {
                    cropBoxes: cropBoxes.map(box => ({ ...box })),
                    layoutType: currentLayoutType
                };
            }

            try {
                updateProgress(0, '开始应用裁剪并进行OCR识别...');
                
                // 创建一个新的数组来存储OCR结果
                const newOcrResults = [];
                
                // 对每一页进行处理
        for (let i = 0; i < processedImages.length; i++) {
                    const image = processedImages[i];
                    updateProgress((i / processedImages.length) * 100, `正在处理第 ${i + 1} 页...`);
                    
                    // 获取当前页面的裁剪框
                    const pageCropBoxes = image.cropSettings?.cropBoxes || [];
                    
                    // 创建一个新的canvas用于处理
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                    const img = new Image();
                
                    // 加载图像
                    await new Promise((resolve, reject) => {
                        img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                            resolve();
                        };
                        img.onerror = reject;
                        img.src = image.dataUrl;
                    });
                    
                    let pageText = '';
                    let totalConfidence = 0;
                    let recognizedCount = 0;
                    
                    // 如果有裁剪框，则按顺序处理每个区域
                    if (pageCropBoxes.length > 0) {
                        // 存储每个区域的OCR结果
                        const areaResults = [];
                        
                        // 按照框选顺序处理每个区域
                        for (let j = 0; j < pageCropBoxes.length; j++) {
                            const box = pageCropBoxes[j];
                            updateProgress(
                                ((i * 100) + (j / pageCropBoxes.length) * 100) / processedImages.length,
                                `正在处理第 ${i + 1} 页的第 ${j + 1} 个区域...`
                            );
                            
                            // 裁剪当前区域
                            const croppedCanvas = document.createElement('canvas');
                            const croppedCtx = croppedCanvas.getContext('2d');
                            croppedCanvas.width = box.width;
                            croppedCanvas.height = box.height;
                            
                            // 从原始图像中裁剪指定区域
                            croppedCtx.drawImage(canvas, 
                                box.x, box.y, box.width, box.height,
                                0, 0, box.width, box.height
                            );
                            
                            // 对裁剪后的区域进行OCR识别
                            const result = await performOCR({
                                dataUrl: croppedCanvas.toDataURL('image/png'),
                                width: box.width,
                                height: box.height
                            });
                            
                            if (result.text && result.text.trim()) {
                                // 保存OCR结果和框选顺序
                                areaResults.push({
                                    text: result.text,
                                    confidence: result.confidence,
                                    index: j
                                });
                                totalConfidence += result.confidence;
                                recognizedCount++;
                            }
                        }
                        
                        // 所有区域OCR完成后，按框选顺序排序并合并文本
                        areaResults.sort((a, b) => a.index - b.index);
                        pageText = areaResults.map(result => result.text).join('\n\n');
                        
                        // 在控制台输出原始OCR合并结果，方便开发者查看
                        console.log(`第 ${i + 1} 页原始OCR合并结果:`, {
                            pageIndex: i + 1,
                            rawOcrText: pageText,
                            areaCount: areaResults.length,
                            confidence: recognizedCount > 0 ? totalConfidence / recognizedCount : 0
                        });
                    } else {
                        // 如果没有裁剪框，直接识别整页
                        updateProgress(
                            (i * 100) / processedImages.length,
                            `正在识别第 ${i + 1} 页（整页识别）...`
                        );
                        
                        const result = await performOCR({
                    dataUrl: canvas.toDataURL('image/png'),
                    width: canvas.width,
                    height: canvas.height
                });
                
                        if (result.text && result.text.trim()) {
                            pageText = result.text;
                            totalConfidence = result.confidence;
                            recognizedCount = 1;
                            
                            // 在控制台输出原始OCR结果，方便开发者查看
                            console.log(`第 ${i + 1} 页原始OCR结果:`, {
                                pageIndex: i + 1,
                                rawOcrText: pageText,
                                confidence: result.confidence
                            });
                        }
                    }
                    
                    // 如果有识别出文本，进行AI优化
                    if (pageText.trim()) {
                        updateProgress(
                            ((i + 0.9) * 100) / processedImages.length,
                            `正在优化第 ${i + 1} 页的文本...`
                        );
                        
                        // 保存原始OCR文本到结果对象中
                        const originalText = pageText;
                        
                        const optimizedResult = await optimizeWithDeepSeek(pageText);
                        newOcrResults[i] = {
                            text: optimizedResult,
                            confidence: recognizedCount > 0 ? totalConfidence / recognizedCount : 0,
                            isOptimized: true,
                            originalText: originalText // 将原始文本保存到结果对象
                        };
                        
                        // 输出优化前后的对比
                        console.log(`第 ${i + 1} 页 AI优化对比:`, {
                            pageIndex: i + 1,
                            original: originalText,
                            optimized: optimizedResult
                        });
        } else {
                        newOcrResults[i] = {
                            text: '',
                            confidence: 0,
                            isOptimized: false,
                            originalText: '' // 即使没有文本，也添加originalText字段保持一致性
                        };
                    }
                    
                    // 每页处理完成后立即更新显示结果
                    displayPreviewAndResults(currentFiles, newOcrResults);
                }
                
                // 更新OCR结果
                ocrResults = newOcrResults;
                
                // 启用下载按钮
                if (downloadTxtBtn) downloadTxtBtn.disabled = false;
                if (downloadDocBtn) downloadDocBtn.disabled = false;
                const downloadOriginalDocBtn = document.getElementById('downloadOriginalDocBtn');
                if (downloadOriginalDocBtn) downloadOriginalDocBtn.disabled = false;
                
                updateProgress(100, 'OCR识别完成');
    } catch (error) {
                console.error('OCR处理失败:', error);
                updateProgress(0, 'OCR处理失败: ' + error.message);
            }
        });
    }

    // 取消裁剪按钮事件
    if (cancelCropBtn) {
cancelCropBtn.addEventListener('click', () => {
            if (cropModal) {
    cropModal.style.display = 'none';
            }
        });
    }

    // 裁剪设置按钮事件
    if (cropSettingsBtn) {
        cropSettingsBtn.addEventListener('click', () => {
            if (processedImages && processedImages.length > 0) {
                if (cropModal) {
                    cropModal.style.display = 'block';
                    showCropPreview(processedImages[currentPageIndex]);
                }
            } else {
                alert('请先上传文件');
            }
        });
    }
    
    // 文件输入事件
    if (fileInput) {
        fileInput.addEventListener('change', async (e) => {
            try {
                const files = e.target.files;
                if (!files || files.length === 0) {
                    console.log('未选择文件');
                    return;
                }
                
                currentFiles = Array.from(files);
                processedImages = [];
                ocrResults = [];
                currentPageIndex = 0;
                cropBoxes = [];
                
                updateProgress(0, '开始处理文件...');
                
                for (const file of currentFiles) {
                    try {
                        let images;
                        
                        if (file.type === 'application/pdf') {
                            images = await processPDF(file);
                        } else if (file.type.startsWith('image/')) {
                            images = [await processImage(file)];
                        } else {
                            throw new Error(`不支持的文件类型: ${file.type}`);
                        }
                        
                        processedImages.push(...images);
                        displayPreviewAndResults(currentFiles, []);
                        
                        // 打开裁剪设置对话框
                        if (cropModal && processedImages.length > 0) {
                            cropModal.style.display = 'block';
                            if (file.type === 'application/pdf') {
                                initPDFNavigation(processedImages.length);
                            }
                            showCropPreview(processedImages[0]);
                        }
                    } catch (error) {
                        console.error(`文件处理失败: ${file.name}`, error);
                        updateProgress(0, `文件处理失败: ${file.name}`, error.message);
                    }
                }
                
                updateProgress(100, '文件处理完成，请设置裁剪区域');
            } catch (error) {
                console.error('处理文件过程中出错:', error);
                updateProgress(0, '处理失败，请重试', error.message);
            }
        });
    }
    
    // 下载TXT按钮事件
    if (downloadTxtBtn) {
        downloadTxtBtn.addEventListener('click', () => {
            try {
                const blob = generateTextFile(ocrResults);
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'ocr_result.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error('生成文本文件失败:', error);
                updateProgress(0, '生成文本文件失败: ' + error.message);
            }
        });
    }
    
    // 下载Word按钮事件
    if (downloadDocBtn) {
        downloadDocBtn.addEventListener('click', generateWordFile);
    }
    
    // 下载原始Word按钮事件
    const downloadOriginalDocBtn = document.getElementById('downloadOriginalDocBtn');
    if (downloadOriginalDocBtn) {
        downloadOriginalDocBtn.addEventListener('click', generateOriginalWordFile);
    }
}

// 在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    initializeButtons();
        initCropCanvasEvents();
});

/**
 * 显示优化详情对话框
 * @param {Object} optimizationDetails - 优化详情
 */
function showOptimizationDetails(optimizationDetails) {
    // 创建模态框
    const detailsModal = document.createElement('div');
    detailsModal.className = 'modal';
    detailsModal.style.display = 'block';
    
    const modalContent = document.createElement('div');
    modalContent.className = 'modal-content';
    modalContent.style.maxWidth = '600px';
    
    const title = document.createElement('h2');
    title.textContent = 'OCR智能优化详情';
    
    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn';
    closeBtn.textContent = '关闭';
    closeBtn.style.backgroundColor = '#f44336';
    closeBtn.style.float = 'right';
    closeBtn.style.marginTop = '-40px';
    
    closeBtn.addEventListener('click', () => {
        document.body.removeChild(detailsModal);
    });
    
    const detailsContent = document.createElement('div');
    detailsContent.innerHTML = `
        <div style="margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #4CAF50;">
            <p><strong>优化方法:</strong> ${optimizationDetails.method === 'ocr_params' ? 'OCR参数优化' : 'AI文本修正'}</p>
            <p><strong>原始置信度:</strong> ${optimizationDetails.originalConfidence.toFixed(2)}%</p>
            <p><strong>优化后置信度:</strong> ${optimizationDetails.confidence.toFixed(2)}%</p>
            <p><strong>提升:</strong> +${(optimizationDetails.confidence - optimizationDetails.originalConfidence).toFixed(2)}%</p>
        </div>
        
        <div style="margin: 10px 0;">
            <h3>检测到的问题:</h3>
            <ul style="padding-left: 20px;">
                ${optimizationDetails.errors.map(err => `<li>${err}</li>`).join('')}
            </ul>
        </div>
        
        <div style="margin: 10px 0;">
            <h3>优化参数:</h3>
            <pre style="background: #f1f1f1; padding: 10px; overflow-x: auto; font-size: 12px;">${JSON.stringify(optimizationDetails.params, null, 2)}</pre>
        </div>
    `;
    
    modalContent.appendChild(title);
    modalContent.appendChild(closeBtn);
    modalContent.appendChild(detailsContent);
    detailsModal.appendChild(modalContent);
    
    document.body.appendChild(detailsModal);
}

/**
 * 更新进度条和状态信息
 * @param {number} progress - 进度值（0-100）
 * @param {string} status - 状态文本
 * @param {string} [error] - 错误信息（如果有）
 */
function updateProgress(progress, status, error = null) {
    const progressBar = document.getElementById('progressBar');
    const progressStatus = document.getElementById('progressStatus');
    
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
    }
    
    if (progressStatus) {
        progressStatus.textContent = status;
        if (error) {
            console.error('处理错误:', error);
            progressStatus.style.color = '#e74c3c';
        } else {
            progressStatus.style.color = '#666';
        }
    }
}

/**
 * 生成原始OCR结果的Word文档
 */
function generateOriginalWordFile() {
    try {
        // 检查是否有OCR结果
        if (!ocrResults || ocrResults.length === 0) {
            updateProgress(0, '没有可用的OCR结果');
            return;
        }

        // 检查 docx 是否可用
        if (typeof docx === 'undefined') {
            console.error('docx.js未正确加载');
            updateProgress(0, 'Word文档生成失败：docx.js未加载');
            return;
        }

        console.log('开始生成原始OCR结果的Word文档...');
        
        // 创建文档
        const doc = new docx.Document({
            sections: [{
                properties: {},
                children: ocrResults.flatMap((result, index) => {
                    const children = [];
                    
                    if (result.originalText && result.originalText.trim()) {
                        // 分段处理
                        const paragraphs = result.originalText.split(/\n\s*\n/);
                        
                        paragraphs.forEach(para => {
                            if (para.trim()) {
                                // 添加文本段落
                                children.push(new docx.Paragraph({
                                    children: [
                                        new docx.TextRun({
                                            text: para.trim(),
                                            color: "000000"
                                        })
                                    ]
                                }));
                            }
                        });
                        
                        // 如果不是最后一页，添加页码标记
                        if (index < ocrResults.length - 1) {
                            children.push(new docx.Paragraph({
                                children: [
                                    new docx.TextRun({
                                        text: `----------第 ${index + 2} 页----------`,
                                        color: "666666",
                                        size: 20
                                    })
                                ],
                                alignment: docx.AlignmentType.CENTER,
                                spacing: {
                                    before: 240,  // 添加上边距
                                    after: 240    // 添加下边距
                                }
                            }));
                        }
                    }
                    
                    return children;
                })
            }]
        });

        console.log('原始文档对象创建成功，准备生成blob...');

        // 生成并下载文档
        docx.Packer.toBlob(doc).then(blob => {
            console.log('Blob生成成功，准备下载...');
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ocr_original_result.docx';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            updateProgress(100, '原始OCR文档生成完成');
        }).catch(error => {
            console.error('生成原始Word文档blob时出错:', error);
            updateProgress(0, '生成原始Word文档失败: ' + error.message);
        });
    } catch (error) {
        console.error('生成原始Word文档时出错:', error);
        updateProgress(0, '生成原始Word文档失败: ' + error.message);
    }
}
  