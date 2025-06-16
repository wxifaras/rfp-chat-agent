from pydantic import BaseModel, Field
from typing import Optional

class DecisionLog(BaseModel):
    # Required fields - these must be provided
    Details: str = Field(..., description="Details of the RFP")
    Est_TCV: str = Field(..., description="Estimated Total Contract Value")
    Division: str = Field(..., description="Business division responsible")
    Pursuit_Sponsor: str = Field(..., description="Name of the pursuit sponsor")
    CDD_Booking_Date: str = Field(..., description="CDD booking date in YYYY-MM-DD format")
    Resources_Required: str = Field(..., description="Required resources for the project")
    Pursuit_Due_Date: str = Field(..., description="Due date for pursuit in YYYY-MM-DD format")
    Client_Ministry_Name: str = Field(..., description="Name of the client ministry")
    I_have_reviewed_the_Opportunity_Review_Policy_the_pursuit_lead_has_the_following_triggers: str = Field(..., description="Opportunity review policy triggers")
    Deal_type_current: str = Field(..., description="Current deal type")
    Expected_Gross_Margin: str = Field(..., description="Expected gross margin percentage")
    Does_Maximus_Canada_have_the_qualifications: str = Field(..., description="Whether Maximus Canada has required qualifications")
    
    # Optional fields - these can be omitted without validation errors
    Pursuit_Name: Optional[str] = Field(None, description="Name of the pursuit/opportunity")
    Decision_by_CDD: Optional[str] = Field(None, description="Decision made by CDD")
    Link_to_pursuit_files: Optional[str] = Field(None, description="URL to pursuit files")
    Associated_files: Optional[str] = Field(None, description="List of associated files")
    Contract_Term: Optional[str] = Field(None, description="Contract duration")
    Has_the_Opportunity_Review_Policy_been_reviewed: Optional[str] = Field(None, description="Whether opportunity review policy has been reviewed")
    Sign_off_status: Optional[str] = Field(None, description="Current sign-off status")
    Next_Step: Optional[str] = Field(None, description="Next steps in the process")
    Partner_Required: Optional[str] = Field(None, description="Whether a partner is required")
    Conditions: Optional[str] = Field(None, description="Any conditions for the pursuit")
    Pursuit_Lead: Optional[str] = Field(None, description="Name of the pursuit lead")
    Salesforce_Opportunity_URL: Optional[str] = Field(None, description="URL to Salesforce opportunity")
    Modified: Optional[str] = Field(None, description="Last modified timestamp")
    Created: Optional[str] = Field(None, description="Creation timestamp")
    Created_By: Optional[str] = Field(None, description="Email of the creator")
    Modified_By: Optional[str] = Field(None, description="Email of the last modifier")