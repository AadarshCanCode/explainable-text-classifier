import React, { useEffect, useMemo, useState } from 'react';
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
import {
    AlertTriangle,
    Activity,
    BrainCircuit,
    Database,
    Loader2,
    Sparkles,
    Trophy,
    GitCompare,
} from 'lucide-react';
import {
    analyzeText,
    comparePredictions,
    getBenchmarks,
    getHealth,
    getModelInfo,
    getTasks,
} from '../services/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const TASK_ORDER = ['fake_news', 'toxic', 'sentiment'];

const fallbackTasks = {
    fake_news: {
        label: 'Fake News Detection',
        description: 'Classify whether a news-like article appears fake or real.',
        sample_text: 'Reuters reported that the central bank raised interest rates after inflation cooled.',
    },
    toxic: {
        label: 'Toxic Comment Detection',
        description: 'Detect harmful or abusive user comments.',
        sample_text: 'You are an idiot and nobody wants your opinion.',
    },
    sentiment: {
        label: 'Sentiment Analysis',
        description: 'Classify text sentiment into negative, neutral, or positive.',
        sample_text: 'The product is okay overall, but I expected better battery life.',
    },
};

function formatSource(source) {
    return source?.replace('HuggingFace: ', '') || 'Unknown source';
}

function formatMetric(value) {
    return `${(value * 100).toFixed(2)}%`;
}

export default function ExplainableClassifier() {
    const [task, setTask] = useState('fake_news');
    const [text, setText] = useState('');
    const [selectedModelChoice, setSelectedModelChoice] = useState('best');

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [compareResult, setCompareResult] = useState(null);
    const [error, setError] = useState(null);

    const [bootLoading, setBootLoading] = useState(true);
    const [bootError, setBootError] = useState(null);
    const [health, setHealth] = useState(null);
    const [tasks, setTasks] = useState(fallbackTasks);
    const [modelInfo, setModelInfo] = useState({});
    const [benchmarks, setBenchmarks] = useState({});

    useEffect(() => {
        let isMounted = true;

        async function loadBootData() {
            try {
                const [healthData, taskData, modelData, benchmarkData] = await Promise.all([
                    getHealth(),
                    getTasks(),
                    getModelInfo(),
                    getBenchmarks(),
                ]);

                if (!isMounted) {
                    return;
                }
                setHealth(healthData);
                setTasks(taskData);
                setModelInfo(modelData);
                setBenchmarks(benchmarkData);
            } catch (fetchError) {
                if (!isMounted) {
                    return;
                }
                setBootError('Could not reach backend metadata endpoints. Run backend on port 8000.');
            } finally {
                if (isMounted) {
                    setBootLoading(false);
                }
            }
        }

        loadBootData();
        return () => {
            isMounted = false;
        };
    }, []);

    const taskOptions = TASK_ORDER.map((id) => ({
        id,
        label: tasks[id]?.label || fallbackTasks[id].label,
    }));

    const selectedTask = tasks[task] || fallbackTasks[task];
    const selectedTaskModelInfo = modelInfo[task] || {};
    const selectedBenchmark = benchmarks[task] || { best_model: null, models: [] };

    const modelOptions = useMemo(() => {
        const dynamicOptions = selectedTaskModelInfo.available_models || [];
        const bestModelName = selectedBenchmark.best_model || selectedTaskModelInfo.best_model;
        const bestLabel = dynamicOptions.find((option) => option.model_name === bestModelName)?.model_label;

        return [
            {
                model_name: 'best',
                model_label: bestLabel ? `Best (${bestLabel})` : 'Best (Auto)',
            },
            ...dynamicOptions,
        ];
    }, [selectedTaskModelInfo, selectedBenchmark]);

    const chartData = useMemo(() => {
        if (!result?.explanation?.length) {
            return null;
        }
        const sorted = [...result.explanation].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
        return {
            labels: sorted.map((item) => item.word),
            datasets: [
                {
                    label: 'LIME Feature Weight',
                    data: sorted.map((item) => item.weight),
                    backgroundColor: sorted.map((item) =>
                        item.weight >= 0 ? 'rgba(13, 148, 136, 0.75)' : 'rgba(234, 88, 12, 0.75)'
                    ),
                    borderColor: sorted.map((item) => (item.weight >= 0 ? '#0f766e' : '#c2410c')),
                    borderWidth: 1,
                    borderRadius: 8,
                },
            ],
        };
    }, [result]);

    const chartOptions = useMemo(
        () => ({
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Top Influential Tokens',
                    color: '#18181b',
                    font: { size: 14, weight: 'bold' },
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `Weight: ${context.parsed.x.toFixed(4)}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: 'rgba(24, 24, 27, 0.08)' },
                    ticks: { color: '#3f3f46' },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#3f3f46' },
                },
            },
        }),
        []
    );

    async function handleAnalyze() {
        if (!text.trim()) {
            setError('Please enter text before running analysis.');
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const chosenModel = selectedModelChoice === 'best' ? null : selectedModelChoice;
            const [predictionData, comparisonData] = await Promise.all([
                analyzeText(text, task, chosenModel),
                comparePredictions(text, task),
            ]);
            setResult(predictionData);
            setCompareResult(comparisonData);
        } catch (analyzeError) {
            setError('Prediction failed. Confirm backend is running and fully initialized.');
        } finally {
            setLoading(false);
        }
    }

    function loadExample() {
        setText(selectedTask.sample_text || fallbackTasks[task].sample_text);
        setError(null);
    }

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
                <section className="xl:col-span-4 flex flex-col gap-5">
                    <div className="card-surface p-5">
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">System</p>
                                <h2 className="text-lg font-black mt-1 flex items-center gap-2">
                                    <Activity className="w-5 h-5 text-teal-700" />
                                    Readiness
                                </h2>
                            </div>
                            <span
                                className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                                    health?.status === 'ok' ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                                }`}
                            >
                                {bootLoading ? 'Loading' : health?.status || 'Unknown'}
                            </span>
                        </div>

                        {bootError && (
                            <div className="mt-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex gap-2">
                                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                                <p>{bootError}</p>
                            </div>
                        )}

                        {!bootError && (
                            <div className="mt-4 grid grid-cols-2 gap-3">
                                <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
                                    <p className="text-xs text-zinc-500">Loaded Models</p>
                                    <p className="text-xl font-black text-zinc-900">{health?.loaded_models?.length ?? '-'}</p>
                                </div>
                                <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
                                    <p className="text-xs text-zinc-500">Total Models</p>
                                    <p className="text-xl font-black text-zinc-900">{health?.total_models ?? '-'}</p>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="card-surface p-5">
                        <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Workflow</p>
                        <h2 className="text-lg font-black mt-1 flex items-center gap-2">
                            <BrainCircuit className="w-5 h-5 text-cyan-700" />
                            Analyze Text
                        </h2>

                        <label className="block mt-4 text-sm font-bold text-zinc-700">Use Case</label>
                        <select
                            value={task}
                            onChange={(event) => {
                                setTask(event.target.value);
                                setSelectedModelChoice('best');
                                setResult(null);
                                setCompareResult(null);
                                setError(null);
                            }}
                            className="w-full mt-2 rounded-xl border border-zinc-300 bg-white px-3 py-2.5 text-sm font-medium outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
                        >
                            {taskOptions.map((taskOption) => (
                                <option key={taskOption.id} value={taskOption.id}>
                                    {taskOption.label}
                                </option>
                            ))}
                        </select>

                        <label className="block mt-4 text-sm font-bold text-zinc-700">Model</label>
                        <select
                            value={selectedModelChoice}
                            onChange={(event) => setSelectedModelChoice(event.target.value)}
                            className="w-full mt-2 rounded-xl border border-zinc-300 bg-white px-3 py-2.5 text-sm font-medium outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
                        >
                            {modelOptions.map((option) => (
                                <option key={option.model_name} value={option.model_name}>
                                    {option.model_label}
                                </option>
                            ))}
                        </select>

                        <div className="mt-3 rounded-xl border border-zinc-200 bg-zinc-50 p-3">
                            <p className="text-sm text-zinc-700">{selectedTask.description}</p>
                        </div>

                        <div className="mt-4 flex items-center justify-between gap-2">
                            <label className="text-sm font-bold text-zinc-700">Input Text</label>
                            <button
                                onClick={loadExample}
                                className="text-xs font-bold uppercase tracking-wide text-cyan-700 hover:text-cyan-900"
                            >
                                Load Example
                            </button>
                        </div>
                        <textarea
                            rows={10}
                            value={text}
                            onChange={(event) => {
                                setText(event.target.value);
                                if (error) {
                                    setError(null);
                                }
                            }}
                            onKeyDown={(event) => {
                                if (event.ctrlKey && event.key === 'Enter') {
                                    handleAnalyze();
                                }
                            }}
                            placeholder="Paste a headline, comment, or review text..."
                            className="w-full mt-2 rounded-xl border border-zinc-300 bg-white px-3 py-3 text-sm leading-relaxed font-mono outline-none resize-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
                        />

                        {error && (
                            <div className="mt-3 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex gap-2">
                                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                                <p>{error}</p>
                            </div>
                        )}

                        <button
                            onClick={handleAnalyze}
                            disabled={loading || bootLoading}
                            className="mt-4 w-full rounded-xl bg-zinc-900 text-amber-100 px-4 py-3 text-sm font-bold tracking-wide hover:bg-zinc-800 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                            {loading ? 'Running Analysis...' : 'Run Explainable Inference'}
                        </button>
                    </div>
                </section>

                <section className="xl:col-span-8 flex flex-col gap-5">
                    <div className="card-surface p-5 sm:p-6">
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Model Benchmarks</p>
                                <h2 className="text-lg sm:text-xl font-black mt-1 flex items-center gap-2">
                                    <Trophy className="w-5 h-5 text-amber-600" />
                                    Accuracy Comparison
                                </h2>
                            </div>
                            <p className="text-xs text-zinc-500">Dataset: {formatSource(selectedTaskModelInfo.source)}</p>
                        </div>

                        <div className="mt-4 overflow-x-auto">
                            <table className="w-full text-sm border-separate border-spacing-y-2 min-w-[640px]">
                                <thead>
                                    <tr className="text-left text-zinc-500 text-xs uppercase tracking-[0.12em]">
                                        <th className="px-3">Model</th>
                                        <th className="px-3">Accuracy</th>
                                        <th className="px-3">Precision</th>
                                        <th className="px-3">Recall</th>
                                        <th className="px-3">F1</th>
                                        <th className="px-3">Train Time</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {selectedBenchmark.models?.map((row) => {
                                        const isBest = row.model_name === selectedBenchmark.best_model;
                                        return (
                                            <tr
                                                key={row.model_name}
                                                className={`rounded-xl ${isBest ? 'bg-emerald-50 border border-emerald-200' : 'bg-zinc-50 border border-zinc-200'}`}
                                            >
                                                <td className="px-3 py-2.5 font-semibold text-zinc-900">
                                                    {row.model_label}
                                                    {isBest && <span className="ml-2 text-xs text-emerald-700 font-bold">BEST</span>}
                                                </td>
                                                <td className="px-3 py-2.5 font-mono">{formatMetric(row.accuracy)}</td>
                                                <td className="px-3 py-2.5 font-mono">{formatMetric(row.precision_weighted)}</td>
                                                <td className="px-3 py-2.5 font-mono">{formatMetric(row.recall_weighted)}</td>
                                                <td className="px-3 py-2.5 font-mono">{formatMetric(row.f1_weighted)}</td>
                                                <td className="px-3 py-2.5 font-mono">{row.train_time_seconds.toFixed(2)}s</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div className="card-surface p-5 sm:p-6 min-h-[760px]">
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Output</p>
                                <h2 className="text-xl sm:text-2xl font-black mt-1">Prediction & Explanation</h2>
                            </div>
                            <div className="text-right">
                                <p className="text-xs text-zinc-500">Dataset Source</p>
                                <p className="text-sm font-bold text-zinc-800">{formatSource(selectedTaskModelInfo.source)}</p>
                            </div>
                        </div>

                        {!result && !loading && (
                            <div className="mt-10 border border-dashed border-zinc-300 rounded-2xl bg-zinc-50/60 p-10 text-center">
                                <Database className="mx-auto w-8 h-8 text-zinc-400" />
                                <h3 className="mt-3 text-lg font-bold text-zinc-800">Ready for Analysis</h3>
                                <p className="mt-2 text-sm text-zinc-500">
                                    Run inference to compare all model outputs and inspect token-level explanation.
                                </p>
                            </div>
                        )}

                        {loading && (
                            <div className="mt-10 border border-zinc-200 rounded-2xl bg-zinc-50 p-10 text-center">
                                <Loader2 className="mx-auto w-10 h-10 text-teal-700 animate-spin" />
                                <p className="mt-3 text-sm font-semibold text-zinc-700">Generating prediction and local explanation...</p>
                            </div>
                        )}

                        {result && !loading && (
                            <div className="mt-6 space-y-5 animate-fade-up">
                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                                    <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4">
                                        <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Prediction</p>
                                        <p className="text-2xl font-black text-zinc-900 mt-2">{result.prediction}</p>
                                    </div>
                                    <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4">
                                        <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Confidence</p>
                                        <p className="text-2xl font-black text-zinc-900 mt-2">{(result.confidence * 100).toFixed(1)}%</p>
                                        <div className="w-full h-2 rounded-full bg-zinc-200 mt-3 overflow-hidden">
                                            <div
                                                className="h-full rounded-full bg-gradient-to-r from-teal-500 to-cyan-500"
                                                style={{ width: `${Math.max(3, result.confidence * 100)}%` }}
                                            />
                                        </div>
                                    </div>
                                    <div className="rounded-2xl border border-zinc-200 bg-zinc-50 p-4">
                                        <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Model Used</p>
                                        <p className="text-base font-black text-zinc-900 mt-2 break-words">{result.model_name}</p>
                                    </div>
                                </div>

                                <div className="rounded-2xl border border-zinc-200 p-4">
                                    <p className="text-sm font-bold text-zinc-800 mb-3">Class Probabilities</p>
                                    <div className="space-y-2">
                                        {result.probabilities?.map((item) => (
                                            <div key={item.label}>
                                                <div className="flex items-center justify-between text-sm">
                                                    <span className="font-semibold text-zinc-700">{item.label}</span>
                                                    <span className="font-bold text-zinc-900">{(item.probability * 100).toFixed(1)}%</span>
                                                </div>
                                                <div className="w-full h-2 rounded-full bg-zinc-100 mt-1 overflow-hidden">
                                                    <div
                                                        className="h-full rounded-full bg-zinc-900"
                                                        style={{ width: `${Math.max(1, item.probability * 100)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {compareResult?.predictions?.length > 0 && (
                                    <div className="rounded-2xl border border-zinc-200 p-4">
                                        <p className="text-sm font-bold text-zinc-800 mb-3 flex items-center gap-2">
                                            <GitCompare className="w-4 h-4 text-violet-700" />
                                            Comparative Outputs (Same Input Across Models)
                                        </p>
                                        <div className="space-y-2">
                                            {compareResult.predictions.map((row) => {
                                                const isUsed = row.model_name === result.model_name;
                                                const isBest = row.model_name === compareResult.selected_by_metric;
                                                return (
                                                    <div
                                                        key={row.model_name}
                                                        className={`rounded-xl border p-3 ${
                                                            isUsed
                                                                ? 'border-cyan-300 bg-cyan-50'
                                                                : isBest
                                                                    ? 'border-emerald-300 bg-emerald-50'
                                                                    : 'border-zinc-200 bg-zinc-50'
                                                        }`}
                                                    >
                                                        <div className="flex items-center justify-between gap-2">
                                                            <div>
                                                                <p className="text-sm font-semibold text-zinc-900">{row.model_label}</p>
                                                                <p className="text-xs text-zinc-600">Prediction: {row.prediction}</p>
                                                            </div>
                                                            <div className="text-right">
                                                                <p className="text-sm font-bold text-zinc-900">{(row.confidence * 100).toFixed(1)}%</p>
                                                                <p className="text-[10px] uppercase tracking-wide text-zinc-500">
                                                                    {isUsed ? 'Used' : isBest ? 'Best by F1' : 'Candidate'}
                                                                </p>
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}

                                {chartData && (
                                    <div className="rounded-2xl border border-zinc-200 p-3 h-[320px]">
                                        <Bar data={chartData} options={chartOptions} />
                                    </div>
                                )}

                                <div className="rounded-2xl border border-zinc-200 overflow-hidden">
                                    <div className="px-4 py-3 bg-zinc-50 border-b border-zinc-200 flex items-center justify-between">
                                        <p className="text-sm font-bold text-zinc-800">Interactive LIME Explanation</p>
                                        <p className="text-xs text-zinc-500">Sandboxed iframe</p>
                                    </div>
                                    <iframe
                                        title="LIME explanation"
                                        srcDoc={result.html}
                                        sandbox="allow-scripts"
                                        className="w-full h-[500px] bg-white"
                                    />
                                </div>
                            </div>
                        )}
                    </div>
                </section>
            </div>
        </div>
    );
}
