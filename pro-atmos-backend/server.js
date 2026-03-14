const express = require("express");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json());

let sensorData = {
  mq5: 0,
  mq7: 0,
  temperature: 0,
  humidity: 0,
  aqi: 0
};

// ESP32 will send data here
app.post("/sensor", (req, res) => {
  sensorData = req.body;
  console.log("Received:", sensorData);
  res.json({ status: "ok" });
});

// Website dashboard will read from here
app.get("/sensor", (req, res) => {
  res.json(sensorData);
});
app.get("/", (req, res) => {
  res.send("Air Quality API is running");
});
app.listen(3000, () => {
  console.log("Server running on port 3000");
});