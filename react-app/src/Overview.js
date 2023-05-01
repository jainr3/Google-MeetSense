import React from 'react';
import PropTypes from 'prop-types';
import './Overview.css';

const Overview = ({ title, content }) => {
  return (
    <div className="overview">
      <div className="overview-header">{title}</div>
      <div className="overview-body">
        <div className="overview-body-main">
            {content}
            </div>
        </div>
      </div>
  );
};

Overview.propTypes = {
  title: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
};

export default Overview;

