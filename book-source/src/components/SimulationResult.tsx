import React from 'react';

const SimulationResult = ({ src, alt, caption, description }) => {
  return (
    <div className="simulation-result-container">
      <figure className="simulation-figure">
        <img src={src} alt={alt} className="simulation-image" />
        <figcaption className="simulation-caption">
          <strong>{caption}</strong>
          {description && <p>{description}</p>}
        </figcaption>
      </figure>
    </div>
  );
};

export default SimulationResult;