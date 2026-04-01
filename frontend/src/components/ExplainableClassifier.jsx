import React, { useState, useEffect, useRef } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { analyzeText } from '../services/api';
import { Loader2, AlertCircle, CheckCircle, Info } from 'lucide-react';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const TASKS = [
    { id: 'fake_news', label: 'Fake News Detection' },
    { id: 'toxic', label: 'Toxic Comment Detection' },
    { id: 'sentiment', label: 'Sentiment Analysis' },
];

export default function ExplainableClassifier() {
    const [text, setText] = useState('');
    const [task, setTask] = useState('fake_news');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const limeContainerRef = useRef(null);

    const handleAnalyze = async () => {
        if (!text.trim()) {
            setError("Please enter some text to analyze.");
            return;
        }
        setError(null);
        setLoading(true);
        try {
            const data = await analyzeText(text, task);
            setResult(data);
        } catch (err) {
            setError("Failed to analyze. Ensure the backend is running on port 8000.");
        } finally {
            setLoading(false);
        }
    };

    // Inject LIME HTML cleanly
    useEffect(() => {
        if (result?.html && limeContainerRef.current) {
            limeContainerRef.current.innerHTML = result.html;
        }
    }, [result]);

    const chartData = result ? {
        labels: result.explanation.map(item => item.word),
        datasets: [
            {
                label: 'Feature Weight',
                data: result.explanation.map(item => item.weight),
                backgroundColor: result.explanation.map(item =>
                    item.weight > 0 ? 'rgba(79, 70, 229, 0.7)' : 'rgba(239, 68, 68, 0.7)'
                ),
                borderColor: result.explanation.map(item =>
                    item.weight > 0 ? 'rgb(79, 70, 229)' : 'rgb(239, 68, 68)'
                ),
                borderWidth: 1,
                borderRadius: 4,
            },
        ],
    } : null;

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y', // horizontal bar chart
        plugins: {
            legend: { display: false },
            title: {
                display: true,
                text: 'Top Predictive Features (LIME Weights)',
                font: { size: 16, weight: 'bold' }
            },
            tooltip: {
                callbacks: {
                    label: (context) => `Weight: ${context.parsed.x.toFixed(4)}`
                }
            }
        },
        scales: {
            x: { grid: { color: '#f3f4f6' } },
            y: { grid: { display: false } }
        }
    };

    return (
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Input Section */}
                <div className="lg:col-span-5 flex flex-col gap-6">
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 transition-all hover:shadow-md">
                        <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                            <CheckCircle className="w-5 h-5 text-indigo-500" />
                            Configuration
                        </h2>

                        <div className="mb-4">
                            <label className="block text-sm font-semibold text-slate-700 mb-2">
                                Use Case
                            </label>
                            <select
                                value={task}
                                onChange={(e) => {
                                    setTask(e.target.value);
                                    setResult(null); // Clear previous result when task changes
                                }}
                                className="w-full bg-slate-50 border border-slate-300 text-slate-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-3 transition-colors outline-none"
                            >
                                {TASKS.map(t => (
                                    <option key={t.id} value={t.id}>{t.label}</option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-semibold text-slate-700 mb-2 border-slate-200 pt-2 border-t mt-4">
                                Input Text
                            </label>
                            <textarea
                                rows="8"
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                placeholder="Type or paste text to analyze here..."
                                className="w-full bg-slate-50 border border-slate-300 text-slate-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-3 transition-colors resize-none outline-none"
                            ></textarea>
                        </div>

                        {error && (
                            <div className="mt-4 p-3 bg-red-50 text-red-700 text-sm rounded-lg flex items-start gap-2 border border-red-100">
                                <AlertCircle className="w-5 h-5 shrink-0" />
                                <span>{error}</span>
                            </div>
                        )}

                        <button
                            onClick={handleAnalyze}
                            disabled={loading}
                            className="mt-6 w-full flex items-center justify-center gap-2 text-white bg-indigo-600 hover:bg-indigo-700 focus:ring-4 focus:ring-indigo-300 font-medium rounded-lg text-sm px-5 py-3 transition-colors disabled:opacity-70 disabled:cursor-not-allowed outline-none"
                        >
                            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
                            {loading ? 'Analyzing...' : 'Analyze Text'}
                        </button>
                    </div>

                    <div className="bg-indigo-50 rounded-2xl p-6 border border-indigo-100 text-indigo-900 text-sm">
                        <h3 className="font-bold mb-2 flex items-center gap-2">
                            <Info className="w-4 h-4" />
                            How it works
                        </h3>
                        <p className="opacity-90 leading-relaxed">
                            This application uses Machine Learning classifiers.
                            The <strong>LIME</strong> framework highlights which words most strongly influenced the model's decision, allowing you to interpret the underlying reasoning transparently.
                        </p>
                    </div>
                </div>

                {/* Results Section */}
                <div className="lg:col-span-7">
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 h-full flex flex-col min-h-[500px]">
                        <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
                            <span className="bg-indigo-100 text-indigo-700 w-8 h-8 rounded-full flex items-center justify-center text-sm">✓</span>
                            Analysis Results
                        </h2>

                        {!result && !loading && (
                            <div className="flex-1 flex flex-col items-center justify-center text-slate-400">
                                {/* Fallback pattern graphic instead of external image to ensure no broken links */}
                                <div className="w-24 h-24 border-4 border-dashed border-slate-200 rounded-full flex items-center justify-center mb-4">
                                    <CheckCircle className="w-8 h-8 text-slate-300" />
                                </div>
                                <p className="mt-2 text-center text-sm">Enter text and click analyze to see predictions and explanations.</p>
                            </div>
                        )}

                        {loading && (
                            <div className="flex-1 flex flex-col items-center justify-center text-indigo-500">
                                <Loader2 className="w-10 h-10 animate-spin mb-4" />
                                <p className="animate-pulse">Running inference & generating explanation...</p>
                            </div>
                        )}

                        {result && !loading && (
                            <div className="flex-1 flex flex-col gap-6 animate-in fade-in duration-500">
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
                                        <p className="text-sm text-slate-500 font-medium mb-1">Prediction</p>
                                        <div className="inline-flex items-center px-4 py-1.5 rounded-full text-lg font-bold bg-indigo-100 text-indigo-800 border border-indigo-200">
                                            {result.prediction}
                                        </div>
                                    </div>
                                    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-center">
                                        <p className="text-sm text-slate-500 font-medium mb-1">Confidence</p>
                                        <div className="text-3xl font-black text-slate-800">
                                            {(result.confidence * 100).toFixed(1)}<span className="text-xl text-slate-400 font-medium">%</span>
                                        </div>
                                    </div>
                                </div>

                                {chartData && (
                                    <div className="bg-white border border-slate-200 rounded-xl p-4 h-[300px]">
                                        <Bar data={chartData} options={chartOptions} />
                                    </div>
                                )}

                                <div className="border border-slate-200 rounded-xl overflow-hidden mt-4 shadow-sm">
                                    <div className="bg-slate-50 border-b border-slate-200 px-4 py-3 text-sm font-semibold text-slate-700">
                                        LIME Text Explanation
                                    </div>
                                    <div className="p-4 overflow-x-auto bg-white" ref={limeContainerRef}></div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
