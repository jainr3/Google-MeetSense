import React from "react";
import "./Dashboard.css";
import Card from "./Card.js"

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
          <div className="navbar-section-title">Google MeetSense</div>
        </div>
        <div className="navbar-section">
          <div className="navbar-user">
            <div className="navbar-user-name">John Smith</div>
            <div className="navbar-user-profile"></div>
          </div>
        </div>
      </nav>

      <div className="dashboard-content">
        <div className="dashboard-left">
         <div className="dashboard-left-top">
            <Card
                title="Overview"
                content={
                <>
                    <h4>Team Rise Sync: Design Update</h4>
                </>
                }
            />
            </div>           
            <div className="dashboard-left-middle">
            <Card
                title="Meeting Summary"
                content={
                <>
                    <p>lorum ipsum</p>
                </>
                }
            />
            </div>
            <div className="dashboard-left-bottom">
            <Card
                title="Action Items"
                content={
                <>
                    <p>lorum ipsum</p>
                </>
                }
            />
            </div>
            <button type="button"> Generate Email </button>
        </div>

        <div className="dashboard-right">
          <Card title="Attendees" content={<p>All attendees</p>}/>
          <div className="dashboard-left-middle">
            <Card title="Filler Words" content={<p>8</p>} />
            <Card title="Jargon" content={<p>14</p>} />
          </div>
          <div className="dashboard-left-bottom">
            <Card title="Attendee Punctuality" content={<p>2</p>} />
            <Card title="Meeting Duration" content={<p>5%</p>} />
          </div>
        </div>
      </div>

      </div>

      

    </div>
  );
}

export default Dashboard;
