// src/Dashboard.js
import React from 'react';
import { motion } from 'framer-motion';

const Dashboard = () => {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1 }}
            style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                height: "100vh",
                color: "#2c3e50"
            }}
        >
            <h2 style={{ fontSize: "2rem" }}>AQI Monitoring Dashboard</h2>
            <p>Select a location to begin...</p>

            {/* We will add the map or data inputs here later */}

        </motion.div>
    );
};

export default Dashboard;