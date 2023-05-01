import React from 'react';
import PropTypes from 'prop-types';
import './CardImg.css';

const CardImg = ({ title, content }) => {
  return (
    <div className="card-img">
      <div className="card-img-header">{title}</div>
      <div className="card-img-body">
        <div className="card-img-body-main">
            {content}
            </div>
        </div>
      </div>
  );
};

CardImg.propTypes = {
  title: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
};

export default CardImg;

