// src/App.js
import React, { useState } from 'react'; // Import useState
import './App.css';
import { motion } from 'framer-motion';
import Dashboard from './Dashboard'; // Import the Dashboard component

function App() {
  // 1. Create a state variable to track if we have started
  const [started, setStarted] = useState(false);

  return (
    <>
      {/* 2. Check the state: If started is true, show Dashboard. If false, show Welcome. */}
      {started ? (
        <Dashboard />
      ) : (
        <div className="welcome-container">
          {/* Animated Title */}
          <motion.h1
            className="title"
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
          >
            TECHNEX 2026
          </motion.h1>

          {/* Animated Subtitle with delay */}
          <motion.p
            className="subtitle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 1 }}
          >
            Build the future. Break the limits.
          </motion.p>

          {/* Animated Button */}
          <motion.button
            className="start-btn"
            onClick={() => setStarted(true)} // <--- 3. This Click triggers the switch!
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{
              type: "spring",
              stiffness: 300,
              damping: 20,
              delay: 0
            }}
            // The "Popping" Hover Effect
            whileHover={{
              scale: 1.1,
              background: "linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%)",
              color: "#010779",
              boxShadow: "0px 0px 25px rgba(0, 210, 255, 0.6)",
              textShadow: "0px 1px 2px rgba(0,0,0,0.2)"
            }}
            // The "Clicking Inside" Effect
            whileTap={{
              scale: 0.9,
              boxShadow: "inset 0px 4px 10px rgba(0, 0, 0, 0.2)"
            }}
            style={{
              background: "#00ddff",
              color: "#3a7bd5",
              fontWeight: "bold",
              border: "2px solid #3a7bd5"
            }}
          >
            Check Air Quality
          </motion.button>
        </div>
      )}
    </>
  );
}

export default App;