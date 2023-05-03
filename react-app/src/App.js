import React from 'react';
import { useState, useEffect } from "react";
import Dashboard from './Dashboard';
import { db, storage } from "./firebase";
import { getDocs, collection } from "firebase/firestore";

function App() {
  return (
    <div>
      <Dashboard />
    </div>
  );
}

export default App;

