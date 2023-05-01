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

function Dashboard() {
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
            <button onClick={sendEmail} style={{backgroundColor: '#1B73E8', borderColor: '#1B73E8', borderRadius: 10, fontSize: 20, color:'white', height: 55, width: 150, marginRight: 10, font: 'Roboto'}}>Share</button>
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
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                </>
                }
            />
            <Card title="Action Items" content={
              <>
                  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
              </>
              }
            />
            </div>
        </div>


        <div className="dashboard-right">
          <CardImg title="Attendees" content={<img style={{marginLeft:"auto", marginRight:"auto",display: "flex", justifyContent: 'center'}} src={attendeesImage} alt=''></img>}/>

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

function sendEmail(){
  window.open('mailto:zz753@cornell.edu; kk828@cornell.edu; rj299@cornell.edu; qjc2@cornell.edu?subject=Team Rise Sync: Design Update&body=Body%20goes%20here');
}

export default Dashboard;
