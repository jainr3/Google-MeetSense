import React from 'react';
import PropTypes from 'prop-types';
import './Metrics.css';

const Metrics = ({ title, content }) => {
  return (
    <div className="metrics">
      <div className="metrics-header">{title}</div>
      <div className="metrics-body">
        <div className="metrics-body-main">
            {content}
            </div>
        </div>
      </div>
  );
};

Metrics.propTypes = {
  title: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
};

export default Metrics;

