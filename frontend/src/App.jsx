import React from 'react';
import ExplainableClassifier from './components/ExplainableClassifier';

function App() {
    return (
        <div className="min-h-screen bg-app text-zinc-950">
            <header className="border-b border-zinc-900/10 sticky top-0 z-20 backdrop-blur bg-amber-50/90">
                <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <div className="h-9 w-9 rounded-lg bg-zinc-900 text-amber-100 flex items-center justify-center text-sm font-bold">
                            XA
                        </div>
                        <div>
                            <p className="text-xs uppercase tracking-[0.24em] text-zinc-500">Explainable AI</p>
                            <h1 className="text-xl sm:text-2xl font-black leading-tight">
                                Text Intelligence Studio
                            </h1>
                        </div>
                    </div>
                    <a
                        href="https://github.com/marcotcr/lime"
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs sm:text-sm font-semibold text-zinc-700 hover:text-zinc-950 transition-colors"
                    >
                        LIME Framework
                    </a>
                </div>
            </header>
            <main className="py-8 sm:py-10">
                <ExplainableClassifier />
            </main>
        </div>
    );
}

export default App;
