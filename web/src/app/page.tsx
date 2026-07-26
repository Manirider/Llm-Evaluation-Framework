"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Upload, BarChart3, Settings, Play } from "lucide-react"

export default function Home() {
  const [dataset, setDataset] = useState<File | null>(null)
  const [isEvaluating, setIsEvaluating] = useState(false)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) setDataset(file)
  }

  const [results, setResults] = useState<any>(null)

  const runEvaluation = async () => {
    setIsEvaluating(true)
    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          metrics: ['bleu', 'rouge_l', 'bert_score', 'faithfulness'],
          datasetName: dataset?.name
        })
      })
      const data = await response.json()
      setResults(data)
    } catch (error) {
      console.error('Evaluation failed:', error)
    } finally {
      setIsEvaluating(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-950 dark:to-zinc-900">
      <nav className="border-b bg-white/80 dark:bg-zinc-950/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-8 w-8 text-blue-600" />
              <span className="text-xl font-bold">LLM Eval Framework</span>
            </div>
            <div className="flex gap-4">
              <Button variant="ghost" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid gap-8 md:grid-cols-2">
          <div className="bg-white dark:bg-zinc-900 rounded-xl p-8 shadow-sm border">
            <h2 className="text-2xl font-semibold mb-6">Upload Dataset</h2>
            
            <div className="border-2 border-dashed border-zinc-300 dark:border-zinc-700 rounded-lg p-12 text-center hover:border-blue-500 transition-colors">
              <Upload className="h-12 w-12 mx-auto mb-4 text-zinc-400" />
              <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-4">
                Drag and drop your JSONL or CSV file here
              </p>
              <input
                type="file"
                accept=".jsonl,.csv"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload">
                <Button variant="outline" asChild>
                  <span>Browse Files</span>
                </Button>
              </label>
              {dataset && (
                <p className="mt-4 text-sm text-green-600 dark:text-green-400">
                  Selected: {dataset.name}
                </p>
              )}
            </div>

            <div className="mt-6 space-y-4">
              <h3 className="font-medium">Metrics Configuration</h3>
              <div className="grid grid-cols-2 gap-3">
                {['BLEU', 'ROUGE-L', 'BERTScore', 'Faithfulness', 'Context Relevancy', 'Answer Relevancy'].map((metric) => (
                  <label key={metric} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" defaultChecked className="rounded" />
                    {metric}
                  </label>
                ))}
              </div>
            </div>

            <Button 
              className="w-full mt-6" 
              size="lg"
              onClick={runEvaluation}
              disabled={!dataset || isEvaluating}
            >
              <Play className="h-4 w-4 mr-2" />
              {isEvaluating ? 'Running Evaluation...' : 'Run Evaluation'}
            </Button>
          </div>

          <div className="bg-white dark:bg-zinc-900 rounded-xl p-8 shadow-sm border">
            <h2 className="text-2xl font-semibold mb-6">Results Dashboard</h2>
            
            <div className="space-y-6">
              <div className="p-4 bg-zinc-50 dark:bg-zinc-800 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Overall Score</span>
                  <span className="text-2xl font-bold text-blue-600">
                    {results ? (results.overall_score * 100).toFixed(0) + '%' : '--'}
                  </span>
                </div>
                <div className="h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-blue-600 transition-all" 
                    style={{ width: results ? `${results.overall_score * 100}%` : '0%' }}
                  ></div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-zinc-50 dark:bg-zinc-800 rounded-lg">
                  <p className="text-sm text-zinc-600 dark:text-zinc-400">Samples Evaluated</p>
                  <p className="text-2xl font-bold">{results?.samples_evaluated || '--'}</p>
                </div>
                <div className="p-4 bg-zinc-50 dark:bg-zinc-800 rounded-lg">
                  <p className="text-sm text-zinc-600 dark:text-zinc-400">Pass Rate</p>
                  <p className="text-2xl font-bold">{results ? (results.pass_rate * 100).toFixed(0) + '%' : '--'}</p>
                </div>
              </div>

              <div className="p-4 bg-zinc-50 dark:bg-zinc-800 rounded-lg">
                <p className="text-sm font-medium mb-3">Metric Performance</p>
                <div className="space-y-2">
                  {results ? Object.entries(results.metrics).map(([key, value]: [string, any]) => (
                    <div key={key} className="flex justify-between text-sm">
                      <span className="text-zinc-600 dark:text-zinc-400 capitalize">{key.replace('_', ' ')}</span>
                      <span className={`font-medium ${value.passed ? 'text-green-600' : 'text-red-600'}`}>
                        {value.score.toFixed(2)} {value.passed ? '✓' : '✗'}
                      </span>
                    </div>
                  )) : ['BLEU', 'ROUGE-L', 'BERTScore', 'Faithfulness'].map((metric) => (
                    <div key={metric} className="flex justify-between text-sm">
                      <span className="text-zinc-600 dark:text-zinc-400">{metric}</span>
                      <span className="font-medium">--</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
