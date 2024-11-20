import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

function App() {
  const [priceData, setPriceData] = useState([]);
  const [metricsData, setMetricsData] = useState({});

  useEffect(() => {
    fetch('/api/prices')
      .then(res => res.json())
      .then(data => {
        setPriceData(data);
      });

    fetch('/api/metrics')
      .then(res => res.json())
      .then(data => {
        setMetricsData(data);
      });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Brent Oil Price Data</h1>
        {/* Display metric values */}
        <div>
          <h2>LSTM model metrics</h2>
          <p>MAE: {metricsData.mae}</p>
          <p>RMSE: {metricsData.rmse}</p>
          <p>MAPE: {metricsData.mape}</p>
        </div>
        <h2>Plot of Oil price data over time</h2>
        <ResponsiveContainer width="90%" height={400}>
          <LineChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="Date" angle={-90} dx={-10} />
            <YAxis dataKey="Price" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="Price" stroke="#8884d8" activeDot={{ r: 8 }} />
          </LineChart>
        </ResponsiveContainer>
      </header>
    </div>
  );
}

export default App;