// src/App.js
import React, { useState } from 'react';
import './App.css';
import Dashboard from './Dashboard';
import Welcome from './Welcome'; // 1. Import your brand new Welcome component!

function App() {
  // Create a state variable to track if we have started
  const [started, setStarted] = useState(false);

  return (
    <>
      {/* Check the state: If started is true, show Dashboard. If false, show Welcome. */}
      {started ? (
        <Dashboard />
      ) : (
        /* 2. Replace all the old animation code with this one clean line! */
        <Welcome onStart={() => setStarted(true)} />
      )}
    </>
  );
}

export default App;