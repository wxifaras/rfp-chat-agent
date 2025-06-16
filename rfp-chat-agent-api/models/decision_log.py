from typing import Optional
from pydantic import BaseModel

class DecisionLog(BaseModel):
    #Pursuit_Name: Optional[str] = None
    Decision_by_CDD: Optional[str] = None
    Details: str
    Est_TCV: str
    Division: str
    Pursuit_Sponsor: str
    CDD_Booking_Date: str
    Link_to_pursuit_files: Optional[str] = None
    Associated_files: Optional[str] = None
    Resources_Required: str
    Contract_Term: Optional[str] = None
    Pursuit_Due_Date: str
    Has_the_Opportunity_Review_Policy_been_reviewed: Optional[str] = None
    Client_Ministry_Name: str
    Sign_off_status: Optional[str] = None
    Next_Step: Optional[str] = None
    Partner_Required: Optional[str] = None
    I_have_reviewed_the_Opportunity_Review_Policy_the_pursuit_lead_has_the_following_triggers: str
    Deal_type_current: str
    Expected_Gross_Margin: str
    Does_Maximus_Canada_have_the_qualifications: str
    Conditions: Optional[str] = None
    Pursuit_Lead: Optional[str] = None
    Salesforce_Opportunity_URL: Optional[str] = None
    Modified: Optional[str] = None
    Created: Optional[str] = None
    Created_By: Optional[str] = None
    Modified_By: Optional[str] = None