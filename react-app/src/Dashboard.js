import React from "react";
import "./Dashboard.css";
import Card from "./Card.js"
import CardImg from "./CardImg.js"
import Metrics from "./Metrics.js"
import attendeesImage from ".//attendees.png"
import delta1 from ".//delta1.png"
import delta2 from ".//delta2.png"
import delta3 from ".//delta3.png"
import delta4 from ".//delta4.png"
import Overview from "./Overview";
import { useState, useEffect } from "react";
import { db, storage } from "./firebase";
import { getDocs, collection, doc, getDoc, onSnapshot } from "firebase/firestore";

function Dashboard() {
  const [meetingList, setMeetingList] = useState([]);
  const [theMeeting, setTheMeeting] = useState([]);

  const meetingsCollectionRef = collection(db, "meetings");

  const getMeeting = async () => {
    const docRef = doc(db, "meetings", "89d8b991-d2db-46bb-a6e7-c02161d3e5ac");
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
      console.log("Document data:", docSnap.data());
    } else {
      // docSnap.data() will be undefined in this case
      console.log("No such document!");
    }
  };

  const unsub = onSnapshot(doc(db, "meetings", "89d8b991-d2db-46bb-a6e7-c02161d3e5ab"), (doc) => {
    // console.log("Current data: ", doc.data());
    setTheMeeting(doc.data());
  });

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

  useEffect(() => {
    getMeeting();
  }, [])

  return (

    <div className="dashboard">
      <nav className="left-navbar">
        <div className="left-navbar-top">
          <div className="left-navbar-title">
          </div>
        </div>
        <div className="left-navbar-body">
        </div>
      </nav>

      <div className="dashboard-main">
        <nav className="navbar">
          <div className="navbar-section">
            <div className="navbar-section-title">MeetSense</div>
          </div>
          <div className="navbar-section">
            <div className="navbar-user">
              <button onClick={sendEmail} style={{ backgroundColor: '#1B73E8', borderColor: '#1B73E8', borderRadius: 10, fontSize: 20, color: 'white', height: 55, width: 150, marginRight: 10, font: 'Roboto' }}>Share</button>
              <div className="navbar-user-profile"></div>
            </div>
          </div>
        </nav>

        <div className="dashboard-content">
          <div className="dashboard-left">
            <div className="dashboard-left-top">
              <Overview
                title="Overview"
                content={
                  <>
                    <h2>Team Rise Sync: Design Update</h2>
                    <p>Date: 4/15/32 &emsp;Time: 2:00-3:03PM&emsp; EST Duration: 63 min&emsp; Recurring: No</p>
                  </>
                }
              />
            </div>
            <div className="dashboard-left-middle">
              <Card
                title="Meeting Summary"
                content={
                  <>
                    <div>
                      <p> {theMeeting.summary}</p>
                    </div>
                  </>
                }
              />
              <Card title="Action Items" content={
                <>
                  <div>
                    <h3>Design</h3>
                    <p> {theMeeting.design_action_items}</p>
                    <h3>Engineering</h3>
                    <p> {theMeeting.engineering_action_items}</p>
                    <h3>Product</h3>
                    <p> {theMeeting.product_action_items}</p>
                  </div>
                </>
              }
              />
            </div>
          </div>


          <div className="dashboard-right">
            <CardImg title="Attendees" content={<img style={{ marginLeft: "auto", marginRight: "auto", display: "flex", justifyContent: 'center' }} src={attendeesImage} alt=''></img>} />

            <div className="dashboard-metrics">
              <Metrics title="Attendee Punctuality 🛈" content={
                <>
                  <h1>2 <img src={delta1} alt=''></img></h1>
                  <p>late by 10 min</p>
                </>} />
              <Metrics title="Meeting Duration 🛈" content={
                <>
                  <h1>5% <img src={delta2} alt=''></img> </h1>
                  <p>over by 3 min</p>
                </>
              } />
            </div>
            <div className="dashboard-metrics">
              <Metrics title="Filler Words 🛈" content={<h1>14 <img src={delta3} alt=''></img></h1>} />
              <Metrics title="Jargon Words 🛈" content={<h1>8 <img src={delta4} alt=''></img></h1>} />
            </div>
          </div>
        </div>

      </div>



    </div>
  );
}

function sendEmail() {
  window.open('mailto:zz753@cornell.edu; kk828@cornell.edu; rj299@cornell.edu; qjc2@cornell.edu?subject=Team Rise Sync: Design Update&body={Body%20goes%20here}');
}

export default Dashboard;
