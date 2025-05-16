// script for stellarium
// Press F12 in Stellarium to activate the script system.

// chapter2 star simulation validate
// ra and de in degrees
var ra = 249.2104;
var de = -12.0386;

// boresight aim
var aim = new V3d(
	Math.cos(de * Math.PI/180) * Math.cos(ra * Math.PI/180),
	Math.cos(de * Math.PI/180) * Math.sin(ra * Math.PI/180),
	Math.sin(de * Math.PI/180)
).toVec3d();
StelMovementMgr.setViewDirectionJ2000(aim);

// field of view
StelMovementMgr.setFov(10.0);

// mag limit
StelSkyDrawer.setFlagStarMagnitudeLimit(true);  
StelSkyDrawer.setCustomStarMagnitudeLimit(5.99); 

// ------------
// chapter4 star simulation validate
// ra and de in degrees
var ra = 55.0588;
var de = 49.7205;

// boresight aim
var aim = new V3d(
	Math.cos(de * Math.PI/180) * Math.cos(ra * Math.PI/180),
	Math.cos(de * Math.PI/180) * Math.sin(ra * Math.PI/180),
	Math.sin(de * Math.PI/180)
).toVec3d();
StelMovementMgr.setViewDirectionJ2000(aim);

// field of view
StelMovementMgr.setFov(9.0);

// mag limit
StelSkyDrawer.setFlagStarMagnitudeLimit(true);  
StelSkyDrawer.setCustomStarMagnitudeLimit(5); 
