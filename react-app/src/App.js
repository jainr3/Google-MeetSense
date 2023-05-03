import React from 'react';
import { useState, useEffect } from "react";
import Dashboard from './Dashboard';
import { db, storage } from "./firebase";
import { getDocs, collection } from "firebase/firestore";

function App() {
  const [meetingList, setMeetingList] = useState([]);

  const meetingsCollectionRef = collection(db, "meetings");

  const getMeetingList = async () => {
    try {
      const data = await getDocs(meetingsCollectionRef);
      const filteredData = data.docs.map((doc) => ({
        ...doc.data(),
        id: doc.id,
      }));
      console.log({ filteredData });
      setMeetingList(filteredData);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    getMeetingList();
  }, [])

  return (
    <div>
      <div>
        {meetingList.map((meeting) => (
          <div>
            <h1> {meeting.id}</h1>
            <p>{meeting.summary}</p>
            <p>{meeting.transcript}</p>
          </div>
        ))}
      </div>
      <Dashboard />
    </div>
  );
}

export default App;

