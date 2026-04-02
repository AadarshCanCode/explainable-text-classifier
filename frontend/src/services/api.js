import axios from 'axios';

const runtimeDefaultApiUrl =
    typeof window !== 'undefined'
        ? `${window.location.protocol}//${window.location.hostname}:8000`
        : 'http://localhost:8000';

const API_URL = import.meta.env.VITE_API_URL || runtimeDefaultApiUrl;

const client = axios.create({
    baseURL: API_URL,
    timeout: 30000,
});

export const getHealth = async () => {
    const response = await client.get('/health');
    return response.data;
};

export const getTasks = async () => {
    const response = await client.get('/tasks');
    return response.data;
};

export const getModelInfo = async () => {
    const response = await client.get('/model-info');
    return response.data;
};

export const getBenchmarks = async () => {
    const response = await client.get('/benchmarks');
    return response.data;
};

export const analyzeText = async (text, task, modelName = null) => {
    try {
        const response = await client.post('/predict', {
            text,
            task,
            model_name: modelName,
        });
        return response.data;
    } catch (error) {
        console.error("Error analyzing text:", error);
        throw error;
    }
};

export const comparePredictions = async (text, task) => {
    const response = await client.post('/predict/compare', {
        text,
        task,
    });
    return response.data;
};
