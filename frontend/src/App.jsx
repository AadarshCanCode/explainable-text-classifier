import React from 'react';
import ExplainableClassifier from './components/ExplainableClassifier';

function App() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 text-slate-900">
            <header className="bg-white shadow-sm border-b border-indigo-100 sticky top-0 z-10">
                <div className="max-w-5xl mx-auto py-4 px-4 sm:px-6 lg:px-8 flex items-center justify-between">
                    <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
                        Explainable AI Text Classifier
                    </h1>
                    <a href="https://github.com/marcotcr/lime" target="_blank" rel="noreferrer" className="text-sm font-medium text-slate-500 hover:text-indigo-600 transition-colors">Powered by LIME</a>
                </div>
            </header>
            <main className="py-10">
                <ExplainableClassifier />
            </main>
        </div>
    );
}

export default App;
