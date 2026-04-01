import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const analyzeText = async (text, task) => {
    try {
        const response = await axios.post(`${API_URL}/predict`, {
            text,
            task
        });
        return response.data;
    } catch (error) {
        console.error("Error analyzing text:", error);
        throw error;
    }
};
