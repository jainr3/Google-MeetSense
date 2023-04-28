import React from 'react';
import PropTypes from 'prop-types';
import './Card.css';

const Card = ({ title, content }) => {
  return (
    <div className="card">
      <div className="card-header">{title}</div>
      <div className="card-body">
        <div className="card-body-main">
            {content}
            </div>
      </div>
    </div>
  );
};

Card.propTypes = {
  title: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
};

export default Card;

