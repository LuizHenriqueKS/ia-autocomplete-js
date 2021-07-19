const inputSource = document.getElementById('inputSource');
const inputTest = document.getElementById('inputTest');
const outputLog = document.getElementById('outputLog');
const inputHiddenLayerSize = document.getElementById('inputHiddenLayerSize');
const btnTrain = document.getElementById('btnTrain');
const btnStop = document.getElementById('btnStop');
let model = undefined;
let dict = undefined;
let endIndex = undefined;
let unknownIndex = undefined;
let sequenceLength = undefined;
let logCache = "";
let oldInput = "";
let oldHiddenLayerSize = undefined;

function main() {
    inputTest.onkeyup = (evt)=>{
        predictTest();
    };
    inputSource.value = generateFirstSource();
    outputLog.value = 'Clique em treinar para iniciar.\n';
    setInterval(()=>{
        if (logCache){
            outputLog.value += logCache;
            outputLog.scrollTop = outputLog.scrollHeight + 1000;
            logCache = '';
        }
    }, 1000);
}

async function train(){
    try {
        const input = inputSource.value.split('\r').join('');
        const hiddenLayerSize = parseInt(inputHiddenLayerSize.value);
        sequenceLength = getSequenceLength(input.split('\n'));

        btnTrain.style.display = 'none';
        btnStop.style.display = 'block';

        dict = generateDict(listChars(input));
        log('Dicion\u00e1rio gerado: ' + dict.join(', '));

        if (oldInput !== input || oldHiddenLayerSize !== hiddenLayerSize){
            model = generateModel(sequenceLength, hiddenLayerSize);
            log('Nova rede gerada');
            oldInput = input;
            oldHiddenLayerSize = hiddenLayerSize;
        }

        log('Processando os dados...');
        const dataset = convertLinesToDataset(input.split('\n'), sequenceLength);
        log('xs: ' + dataset.xs.join(', '));
        log('ys: ' + dataset.ys.join(', '));

        log('Testando rede...');
        let test = await predict(dict[2]);
        log("Teste: " + test);

        log('Iniciando treinamento...');
        await fitDataset(model, dataset);
        log('Treinamento conclu\u00eddo');
        
        log('Testando rede...');
        test = await predict(dict[2]);
        log("Teste: " + test);
    } catch (e) {
        log('Error: ' + e.message);
        console.error(e);
    } finally {
        btnStop.style.display = 'none';
        btnTrain.style.display = 'block';
        inputTest.removeAttribute('readonly');
    }
}

function stopTrain(){
    logCache = '';
    model.stopTraining = true;
}

function predict(input){
    const sequence = generateSequence(input, sequenceLength);
    const xs = tf.tensor([sequence]);
    const ys = model.predict(xs);
    const bestIndex = tf.argMax(ys.arraySync()[0][0]).arraySync();
    return dict[bestIndex];
}

function predictTest(){
    inputPredict.value = '';
    let context = inputTest.value;
    let stop = false;
    while (context.length < sequenceLength && !stop){
        const result = predict(context);
        if (result === '<end>') {
            stop = true;
        } else {
            context += result;
        }
    }
    inputPredict.value = context;
}

async function fitDataset(model, dataset){
    const xs = tf.tensor(dataset.xs);
    const ys = tf.tensor(dataset.ys);
    await model.fit(xs, ys, {epochs: 100000, callbacks: [{
        onEpochEnd(epoch, log){
            if (log.acc == 1){
                model.stopTraining = true;
            }
            if (!model.stopTraining){
                logCache += `[\u00c9poca=${epoch} Perda=${log.loss} Precis\u00e3o=${log.acc}]\n`;
            }
        }
    }]});
}

function generateModel(sequenceLength, hiddenLayerSize) {   
    log('Tamanho dos dados de entrada: ' + (sequenceLength));
    const result = tf.sequential();
    result.add(tf.layers.inputLayer({inputShape: [sequenceLength]}));
    result.add(tf.layers.embedding({inputDim: dict.length, outputDim: 4}));
    result.add(tf.layers.flatten());
    result.add(tf.layers.dropout({rate: 0.2}));
    result.add(tf.layers.dense({units: hiddenLayerSize, activation: 'relu'}));
    result.add(tf.layers.dense({units: dict.length, activation: 'softmax'}));
    result.add(tf.layers.reshape({targetShape: [1, dict.length]}));
    result.compile({loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'], optimizer: tf.train.adam()});
    return result;
}

function convertLinesToDataset(lines, sequenceLength){
    const result = {xs:[], ys: []};
    for (const line of lines) {
        if (line){
            const data = convertLineToData(line, sequenceLength);
            result.xs.push(...data.xs);
            result.ys.push(...data.ys);
        }
    }
    return result;
}

function convertLineToData(line, sequenceLength){
    const result = {xs: [], ys:[]};
    let context = '';
    for (const output of line.split('')){
        result.xs.push(generateSequence(context, sequenceLength));
        context += output;
        result.ys.push([[dict.indexOf(output)]]);
    }
	result.xs.push(generateSequence(context, sequenceLength));
	result.ys.push([[endIndex]]);
    return result;
}

function generateSequence(context, sequenceLength){
    const result = context.split('').map(ch=>dict.indexOf(ch))
    for (let i = result.length; i < sequenceLength; i++){
        result.push(endIndex);
    }
    return result;
}

function getSequenceLength(lines){
    return lines.map(line=>line.length).reduce((a,b)=>Math.max(a,b));
}

function listChars(text){
    return text.split('\r').join('').split('\n').join('').split('');
}

function log(text){
    logCache += text + '\n';
}

function generateDict(values){
    const result = ['<unknown>', '<end>'];
    unknownIndex = 0;
    endIndex = 1;
    for (const val of values) {
        if (!result.includes(val)){
            result.push(val);
        }        
    }
    return result;
}

function generateFirstSource() {
    let result = '';
    for (let x = 1; x < 4; x++) {
        for (let y = 1; y < 4; y++) {
            result += `${x}+${y}=${x+y}\n`;
        }
    }
    result = result.split('\n');
    result.splice(2,1)
    result.splice(5,1);
    result = result.join('\n');
    return result;
}

main();