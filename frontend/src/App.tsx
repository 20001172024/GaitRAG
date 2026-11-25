import { Disclaimer } from "./components/Disclaimer";

export default function App() {
  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <header>
        <h1 className="text-3xl font-bold">GaitRAG</h1>
        <p className="text-gray-600">Evidence-grounded gait analysis assistant.</p>
      </header>
      <Disclaimer />
      <p className="text-gray-500">Upload and analysis are coming in later milestones.</p>
    </main>
  );
}
